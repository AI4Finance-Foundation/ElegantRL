import os.path
import time

import numpy as np
import numpy.random as rd
import torch


class DownLinkEnv0:  # stable 2021-12-12
    def __init__(
        self,
        bases_n=4,
        users_n=8,
        power=1.0,
        sigma=1.0,
        csi_noise_var=0.1,
        csi_clip=3.0,
        env_cwd=".",
        if_sum_rate=True,
    ):
        """
        :param bases_n: antennas number of BaseStation
        :param users_n: user number
        :param power: the power of BaseStation. `self.get_action_power()`
        :param sigma: SNR=MaxPower/sigma**2, suggest tp keep power=1.0
        :param csi_noise_var: the noise var of Channel State Information
        """
        self.bases_n = bases_n
        self.users_n = users_n
        self.power = power
        self.sigma = sigma
        self.csi_clip = csi_clip
        self.csi_noise_var = csi_noise_var

        self.env_name = "DownLinkEnv-v0"
        self.env_num = 1
        dir(env_cwd)
        self.state_dim = (2, users_n, bases_n)
        self.action_dim = int(np.prod((2, bases_n, users_n)))
        self.max_step = 2**10
        self.if_discrete = False
        self.target_return = 2.1 * self.max_step

        self.state = None
        self.step_i = None
        self.ur_bs_ary0 = None  # original hall
        self.ur_bs_ary1 = None  # noisy hall (observation hall)
        self.get_rate = (
            get_sum_rate_miso if if_sum_rate else get_min_rate_miso
        )  # SumRate or Max-MinRate

    def reset(self):
        self.ur_bs_ary0 = self.get_randn_complex(
            (self.max_step + 1, self.users_n, self.bases_n)
        ) * np.sqrt(1 / 2)
        ur_bs_noise_ary = self.get_randn_complex(
            (self.max_step + 1, self.users_n, self.bases_n)
        ) * np.sqrt(self.csi_noise_var / 2)
        self.ur_bs_ary1 = self.ur_bs_ary0 + ur_bs_noise_ary

        self.step_i = 0
        return self.get_state()

    def step(self, action):
        reward = self.get_reward(action)  # write before `self.step_i += 1`
        self.step_i += 1
        self.state = self.get_state()  # write after `self.step_i += 1`

        done = self.step_i == self.max_step
        return self.state, reward, done, None

    def get_state(self):  # write after `self.step_i += 1`
        hall_noisy = self.ur_bs_ary1[self.step_i]
        return np.stack((hall_noisy.real, hall_noisy.imag))

    def get_reward(self, action):
        action = action.reshape((2, self.bases_n, self.users_n))
        w = action[0] + action[1] * 1j

        h = self.state
        return self.get_rate(h, w, self.sigma)

    def get_randn_complex(self, size):
        real_part = (
            rd.randn(*size).astype(np.float32).clip(-self.csi_clip, self.csi_clip)
        )
        imag_part = (
            rd.randn(*size).astype(np.float32).clip(-self.csi_clip, self.csi_clip)
        )
        return real_part + imag_part * 1j

    def get_action_norm_power(self, action):
        return action * (self.power / np.linalg.norm(action))


"""utils"""


def get_sum_rate_miso(h, w, sigma=1.0) -> float:
    """
    :param h: hall, (state), h.shape == (users_n, bases_n)
    :param w: weight, (action), w.shape == (bases_n, users_n)
    :param sigma: SNR = Power / sigma**2
    :return: isinstance(rates.sum(), float), rates.shape = (users_n,)
    """

    # channel_gains = np.power(np.abs(np.dot(h, w / sigma**2)), 2)
    # signal_gains = np.diag(channel_gains)
    # interference_gains = channel_gains.sum(axis=1) - signal_gains
    # rates = np.log2(1 + signal_gains / (interference_gains + 1))
    # same as
    channel_gains = np.power(np.abs(np.dot(h, w)), 2)
    signal_gains = np.diag(channel_gains)
    interference_gains = channel_gains.sum(axis=1) - signal_gains
    rates = np.log2(1 + signal_gains / (interference_gains + sigma**2))
    return sum(rates)


def get_min_rate_miso(h, w, sigma=1.0) -> float:  # almost same as `get_sum_rate_miso`
    channel_gains = np.power(np.abs(np.dot(h, w)), 2)
    signal_gains = np.diag(channel_gains)
    interference_gains = channel_gains.sum(axis=1) - signal_gains
    rates = np.log2(1 + signal_gains / (interference_gains + sigma**2))
    return min(rates)  # only different


"""old"""


class DownLinkEnv1:  # [ElegantRL.2021.11.11]
    def __init__(
        self,
        bases_n=4,
        users_n=8,
        power=1.0,
        sigma=1.0,
        csi_noise_var=0.1,
        csi_clip=3.0,
        env_cwd=".",
        curr_schedules=None,
    ):
        """
        :param bases_n: antennas number of BaseStation
        :param users_n: user number
        :param power: the power of BaseStation. `self.get_action_power()`
        :param sigma: SNR=MaxPower/sigma**2, suggest tp keep power=1.0
        :param csi_noise_var: the noise var of CSI (Channel State Information)
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: CWD (current working directory) path for curriculum learning
        :param curr_schedules: [list] the list of schedule value (0.0 ~ 1.0]
        """
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.power = power  #
        self.sigma = sigma  # SNR=15dB = power/sigma**2 =
        self.csi_clip = csi_clip
        self.csi_noise_var = csi_noise_var  # 0.0, 0.1
        # assert self.csi_noise_var == 0

        self.env_name = "DownLinkEnv-v1"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (2, users_n, bases_n)
        self.action_dim = int(2 * bases_n * users_n)
        self.max_step = 2**12
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.ur_bs_ary = None  # Hall
        self.ur_bs_ary_noisy = None  # noisy Hall

        """curriculum learning"""
        # csi_noise_var=0.1, power = 1, sigma=1
        self.reward_offset = np.array(
            [
                [1.0, 2.106],
                [2.0, 2.461],
                [3.0, 2.321],
                [4.0, 2.017],
                [5.0, 1.964],
                [6.0, 2.035],
                [7.0, 2.137],
                [8.0, 2.197],
            ]
        )[users_n - 1, 1]
        self.curr_target_return = self.reward_offset - self.reward_offset
        self.curr_schedules = (
            (
                [
                    0.01,
                    0.05,
                    0.10,
                ]
                + list(np.linspace(0.2, 1.0, 10 - 1)[:-1])
            )
            if curr_schedules is None
            else curr_schedules
        )
        self.curr_schedule = self.curr_schedules[0]

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{self.curr_schedule}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_ur_bs_ary = np_save_dict["fixed_ur_bs"]
        else:
            print(f"DownLinkEnv-v1: FileNotFound {self.npy_data_path}")

    def save_fixed_data_in_disk(self):
        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        self.fixed_ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        np_save_dict = {"fixed_ur_bs": self.fixed_ur_bs_ary}
        np.savez(self.npy_data_path, **np_save_dict)
        print(f"DownLink-v1: {self.npy_data_path}")

    def reset(self):
        # Hall: ur_bs
        self.ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # Hall:
        noise = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((self.csi_noise_var / 2) ** 0.5)
        self.ur_bs_ary_noisy = self.ur_bs_ary + noise
        # assert self.csi_noise_var == 0
        # self.ur_bs_ary_noisy = self.ur_bs_ary

        """curriculum learning"""
        # self.curr_schedule = self.get_curr_schedule()  # todo temp
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)
        self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary_noisy)
        # assert self.csi_noise_var == 0

        self.step_i = 0
        return self.get_state()  # state

    def step(self, action):
        reward = self.get_reward(action)

        self.step_i += 1  # write before `self.get_state()`
        state = self.get_state()

        done = self.step_i == self.max_step
        return state, reward, done, None

    def get_state(self):
        hall_noisy = self.ur_bs_ary_noisy[self.step_i]
        return np.stack((hall_noisy.real, hall_noisy.imag))

    def get_reward(self, action):  # get_sum_rate we need a sigma here(maybe).
        h = self.ur_bs_ary[self.step_i]

        action = action.reshape((2, self.bases_n, self.users_n))
        w = action[0] + action[1] * 1j
        return get_sum_rate_miso(h, w, self.sigma) - self.reward_offset

    def get_action_norm_power(self, action=None):
        # action2 = action.reshape((2, -1))
        # action_normal = np.sqrt(np.power(action2, 2).sum(axis=0, keepdims=True)) * np.sqrt(self.power)
        # return (action2 / action_normal).reshape(-1)  # action
        return action / (np.linalg.norm(action) * self.power)

    def get_action_mmse(self, state):  # not-necessary
        # (state, bases_n, users_n, power, csi_noise_var)
        h_noisy = state[0] + state[1] * 1j
        w_mmse = func_mmse(
            h_noisy, self.bases_n, self.users_n, self.power, self.csi_noise_var
        )
        return np.stack((w_mmse.real, w_mmse.imag))

    def get_curr_schedule(self):
        with open(self.curr_txt_path) as f:
            curr_schedule = float(eval(f.readlines()[-1]))
        assert 0.0 <= curr_schedule <= 1.0
        return curr_schedule

    def curriculum_learning_for_evaluator(self, r_avg):
        """Call this function in `Class Evaluator.evaluate_and_save()`
        if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
            self.eval_env.curriculum_learning_for_evaluator(r_avg)
        """
        if r_avg < self.curr_target_return or self.curr_schedule == 1:
            return

        """update curriculum learning tau in disk"""
        self.curr_schedule = self.curr_schedules.pop(0) if self.curr_schedules else 1.0
        with open(self.curr_txt_path, "w+") as f:
            f.write(f"{self.curr_schedule}\n")

        print(
            f"\n| DownLinkEnv: {self.curr_schedule:8.3f}    "
            f"currR {self.curr_target_return:8.3f}    avgR {r_avg:8.3f}"
        )

    def get_random_from_fixed_array(self, fixed_ary, rd_ary):  # [ElegantRL.2011.11.24]
        rd_ary[:] = rd_ary * np.sqrt(self.curr_schedule) + fixed_ary * np.sqrt(
            1 - self.curr_schedule
        )


class DownLinkEnv2(DownLinkEnv1):  # [ElegantRL.2021.11.11]
    def __init__(
        self,
        bases_n=4,
        users_n=6,
        power=1.0,
        sigma=1.0,
        csi_noise_var=0.0,
        csi_clip=3.0,
        env_cwd=".",
        curr_schedules=None,
    ):
        """
        :param bases_n: antennas number of BaseStation
        :param users_n: user number
        :param power: the power of BaseStation. `self.get_action_power()`
        :param sigma: SNR=MaxPower/sigma**2, suggest tp keep power=1.0
        :param csi_noise_var: the noise var of CSI (Channel State Information)
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: [str] CWD (current working directory) path for curriculum learning
        :param curr_schedules: [list] the list of schedule value (0.0 ~ 1.0]
        """
        super(DownLinkEnv1).__init__()
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.power = power
        assert self.power == 1.0
        self.sigma = sigma
        self.csi_clip = csi_clip
        self.csi_noise_var = csi_noise_var  # 0.0, 0.1
        assert self.csi_noise_var == 0

        self.env_name = "DownLinkEnv-v2"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (2, users_n, bases_n)
        self.action_dim = int(2 * bases_n * users_n)
        self.max_step = 2**12
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.ur_bs_ary = None  # Hall
        self.ur_bs_ary_noisy = None  # noisy Hall

        """curriculum learning"""
        # csi_noise_var=0.0, power = 1, sigma=1
        self.reward_offset = np.array(
            [
                [1.0, 2.424],
                [2.0, 2.906],
                [3.0, 2.794],
                [4.0, 2.339],
                [5.0, 2.385],
                [6.0, 2.547],
                [7.0, 2.652],
                [8.0, 2.713],
            ]
        )[users_n - 1, 1]
        self.curr_target_return = self.reward_offset - self.reward_offset
        self.curr_schedules = (
            (
                [
                    0.01,
                    0.05,
                    0.10,
                ]
                + list(np.linspace(0.20, 1.00, 20 - 1)[:-1])
            )
            if curr_schedules is None
            else curr_schedules
        )

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{self.curr_schedules[0]}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_ur_bs_ary = np_save_dict["fixed_ur_bs"]
        else:
            print(f"DownLinkEnv: FileNotFound {self.npy_data_path}")

    def reset(self):
        # Hall: ur_bs
        self.ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # Hall:
        # noise = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #          .clip(-self.csi_clip, self.csi_clip) +
        #          rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #          .clip(-self.csi_clip, self.csi_clip) * 1j
        #          ) * np.array((self.csi_noise_var / 2) ** 0.5)
        # self.ur_bs_ary_noisy = self.ur_bs_ary + noise
        # assert self.csi_noise_var == 0
        self.ur_bs_ary_noisy = self.ur_bs_ary

        """curriculum learning"""
        self.curr_schedule = self.get_curr_schedule()
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)
        # self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary_noisy)
        # assert self.csi_noise_var == 0

        self.step_i = 0
        return self.get_state()  # state


class DownLinkEnv3(DownLinkEnv1):
    def __init__(
        self,
        bases_n=5,
        users_n=4,
        relay_n=20,
        power=0.2,
        csi_clip=3.0,
        env_cwd=".",
        curr_delta=1 / 256,
        curr_schedule_init=1 / 32,
    ):
        """
        :param bases_n: antennas number of BS (BaseStation)
        :param users_n: user number
        :param relay_n: antennas number of ReLay IRS (Intelligent Reflecting Surfaces)
        :param power: the power of BaseStation. `self.get_action_power()`
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: CWD (current working directory) path for curriculum learning
        :param curr_delta: the delta of schedule value (0.0 ~ 1.0) for curriculum learning
        :param curr_schedule_init: the initialization value of schedule value (0.0 ~ 1.0)
        """
        super(DownLinkEnv1).__init__()
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.relay_n = relay_n  # rl
        self.power = power
        self.csi_clip = csi_clip
        self.reward_offset = 15.0

        self.env_name = "DownLinkEnv-v3"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (2, users_n + bases_n, relay_n + bases_n)
        self.action_dim = int(relay_n * 2)
        self.max_step = 2**10
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.bs_rl_ary = self.bs_rl = None
        self.ur_rl_ary = self.ur_rl = None
        self.ur_bs_ary = self.ur_bs = None

        """curriculum learning"""
        self.curr_delta = curr_delta
        self.curr_target_return = 18 - self.reward_offset  # 18.4

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{curr_schedule_init}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_bs_rl_ary = np_save_dict["fixed_bs_rl"]
            self.fixed_ur_rl_ary = np_save_dict["fixed_rl_ur"]
            self.fixed_ur_bs_ary = np_save_dict["fixed_ur_bs"]

    def save_fixed_data_in_disk(self):
        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        self.fixed_ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        self.fixed_bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        self.fixed_ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        self.fixed_ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 5) ** 0.5)

        np_save_dict = {
            "fixed_bs_rl": self.fixed_bs_rl_ary,
            "fixed_rl_ur": self.fixed_ur_rl_ary,
            "fixed_ur_bs": self.fixed_ur_bs_ary,
        }
        np.savez(self.npy_data_path, **np_save_dict)
        print(f"DownLink-v2: {self.npy_data_path}")

    def reset(self):
        # G: bs_rl
        self.bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # H_irs: ur_rl
        self.ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # H_dk: ur_bs
        self.ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 5) ** 0.5)

        """curriculum learning"""
        self.curr_schedule = self.get_curr_schedule()
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_bs_rl_ary, self.bs_rl_ary)
        self.get_random_from_fixed_array(self.fixed_ur_rl_ary, self.ur_rl_ary)
        self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)

        self.step_i = 0
        return self.get_state()  # state

    def get_state(self):
        # self.state_dim = (2, users_n + bases_n, relay_n + bases_n)
        state = np.zeros(self.state_dim, dtype=np.float32)

        # G: bs_rl
        self.bs_rl = self.bs_rl_ary[self.step_i, : self.relay_n]
        state[0, : self.bases_n, : self.relay_n] = self.bs_rl.real
        state[1, : self.bases_n, : self.relay_n] = self.bs_rl.imag

        # H_irs: ur_rl
        self.ur_rl = self.ur_rl_ary[self.step_i, : self.relay_n]
        state[0, self.bases_n :, : self.relay_n] = self.ur_rl.real
        state[1, self.bases_n :, : self.relay_n] = self.ur_rl.imag
        self.ur_rl = self.ur_rl.transpose()

        # H_dk: ur_bs
        self.ur_bs = self.ur_bs_ary[self.step_i, : self.relay_n]
        state[0, self.bases_n :, self.relay_n :] = self.ur_bs.real
        state[1, self.bases_n :, self.relay_n :] = self.ur_bs.imag
        self.ur_bs = self.ur_bs.transpose()
        return state

    def get_reward(self, action):
        phi = action[: self.relay_n] + action[self.relay_n :] * 1j

        temp_list = []
        j_opt = 0
        for i in range(self.users_n):
            h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
            h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)

            variable = (self.bs_rl * phi).dot(h_irs_i) + h_dk_i
            variable = variable.reshape(self.bases_n, 1)
            variable_h = variable.transpose().conjugate()

            temp_list.append((variable, variable_h))

            j_opt += self.power * (variable.dot(variable_h))
        j_opt += np.eye(self.bases_n)

        reward = 0
        for i in range(self.users_n):
            variable, variable_h = temp_list[i]

            e_mmse = 1 - self.power * variable_h.dot(np.linalg.inv(j_opt)).dot(variable)
            reward += float(np.real(np.log2(e_mmse) * -1))
        return reward - self.reward_offset

    def get_iter_action(self, iter_n=32):
        random_phi = np.random.normal(0, 1, size=(self.relay_n,)).clip(-1, 1)
        phi_ite_mat = np.diag(np.exp(1j * np.pi * random_phi))
        for k in range(iter_n):
            j_random = 0
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i) + h_dk_i
                """------------------------------------------------------------"""
                # self.var_i_mid_list_phi.append(var_i_mid_list)
                """------------------------------------------------------------"""
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                j_random += self.power * (var_i_mid_ite.dot(var_i_mid_ite_h))
            j_random += np.eye(self.bases_n)

            a_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            b_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            c_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            d_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i) + h_dk_i
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                """------------------------------------------------------------"""
                # h_irs_i = self.h_irs[:, i].reshape(self.irs_num, 1)
                # var_i_mid_ite = self.var_i_mid_list_phi[i].reshape(self.base_num, 1)
                # var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()         # 用列表加载需要每次清空列表
                """------------------------------------------------------------"""
                w_i_mat = (np.sqrt(self.power) * np.linalg.inv(j_random)).dot(
                    var_i_mid_ite
                )
                w_i_mat_h = w_i_mat.transpose().conjugate()
                e_i = np.real(
                    1
                    - self.power
                    * (var_i_mid_ite_h.dot(np.linalg.inv(j_random)).dot(var_i_mid_ite))
                )
                a_mat += (1 / e_i) * self.bs_rl.transpose().conjugate().dot(
                    w_i_mat
                ).dot(w_i_mat_h).dot(self.bs_rl)
                h_irs_i_h = h_irs_i.transpose().conjugate()
                b_mat += self.power * h_irs_i.dot(h_irs_i_h)
                d_mat += (1 / e_i) * (
                    np.sqrt(self.power) * h_irs_i.dot(w_i_mat_h).dot(self.bs_rl)
                )
                for m in range(self.users_n):
                    h_irs_m = self.ur_rl[:, m].reshape(self.relay_n, 1)
                    h_dk_m = self.ur_bs[:, m].reshape(self.bases_n, 1)
                    h_dk_m_h = h_dk_m.transpose().conjugate()
                    c_mat += (
                        self.power
                        * (1 / e_i)
                        * h_irs_m.dot(h_dk_m_h)
                        .dot(w_i_mat)
                        .dot(w_i_mat_h)
                        .dot(self.bs_rl)
                    )
            # self.var_i_mid_list_phi = []
            psi = a_mat * b_mat.transpose()
            v = np.diag(c_mat - d_mat)
            eig_val, eig_vct = np.linalg.eig(psi)
            lamba_max = np.max(np.real(eig_val))
            phi_ite = np.diagonal(phi_ite_mat)  # get the eye element
            last_rt = 0
            for ite in range(300):
                phi_ite = np.exp(
                    1j
                    * np.angle(
                        (lamba_max * (np.eye(self.relay_n)) - psi).dot(phi_ite)
                        - v.conjugate()
                    )
                )
                rt = np.real(
                    phi_ite.transpose().conjugate().dot(psi).dot(phi_ite)
                ) + 2 * np.real(phi_ite.transpose().conjugate().dot(v.conjugate()))
                if ite >= 20 and (
                    abs(rt - last_rt) / abs(rt + 0.00000000001) <= 0.0001
                ):
                    phi_ite_mat = np.diag(phi_ite)
                    break
                last_rt = rt
        action = phi_ite_mat.sum(axis=0)
        action = np.hstack((action.real, action.imag))
        return action


class DownLinkEnv4(DownLinkEnv1):
    def __init__(
        self,
        bases_n=3,
        users_n=3,
        relay_n=12,
        power=0.2,
        csi_clip=3.0,
        env_cwd=".",
        curr_delta=0.005,
        curr_schedule_init=0.05,
    ):
        """
        :param bases_n: antennas number of BS (BaseStation)
        :param users_n: user number
        :param relay_n: antennas number of ReLay IRS (Intelligent Reflecting Surfaces)
        :param power: the power of BaseStation. `self.get_action_power()`
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: CWD (current working directory) path for curriculum learning
        :param curr_delta: the delta of schedule value (0.0 ~ 1.0) for curriculum learning
        :param curr_schedule_init: the initialization value of schedule value (0.0 ~ 1.0)
        """
        super(DownLinkEnv1).__init__()
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.relay_n = relay_n  # rl
        self.power = power
        self.csi_clip = csi_clip
        self.reward_offset = {
            8: (07.1 + 4.5) / 2,  # user=3, bs=3, relay= 8
            12: (09.8 + 5.4) / 2,  # user=3, bs=3, relay= 8
            16: (11.7 + 6.0) / 2,  # user=3, bs=3, relay= 8
        }[relay_n]
        assert users_n == 3 and bases_n == 3

        self.env_name = "DownLinkEnv-v4"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (
            2,
            users_n + bases_n,
            relay_n,
        )  # (2, users_n + bases_n, relay_n + bases_n)
        self.action_dim = int(relay_n * 2)
        self.max_step = 2**12
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.bs_rl_ary = self.bs_rl = None
        self.ur_rl_ary = self.ur_rl = None
        # self.ur_bs_ary = self.ur_bs = None

        """curriculum learning"""
        self.curr_delta = curr_delta
        self.curr_target_return = self.reward_offset - self.reward_offset  # 18.4

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{curr_schedule_init}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_bs_rl_ary = np_save_dict["fixed_bs_rl"]
            self.fixed_ur_rl_ary = np_save_dict["fixed_rl_ur"]
            # self.fixed_ur_bs_ary = np_save_dict['fixed_ur_bs']

    def save_fixed_data_in_disk(self):
        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        self.fixed_bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        self.fixed_ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        # self.fixed_ur_bs_ary = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                         .clip(-self.csi_clip, self.csi_clip) +
        #                         rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                         .clip(-self.csi_clip, self.csi_clip) * 1j
        #                         ) * np.array((1 / 5) ** 0.5)

        np_save_dict = {
            "fixed_bs_rl": self.fixed_bs_rl_ary,
            "fixed_rl_ur": self.fixed_ur_rl_ary,
            # 'fixed_ur_bs': self.fixed_ur_bs_ary,
        }
        np.savez(self.npy_data_path, **np_save_dict)
        print(f"DownLink-v2: {self.npy_data_path}")

    def reset(self):
        # G: bs_rl
        self.bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # H_irs: ur_rl
        self.ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # # H_dk: ur_bs
        # self.ur_bs_ary = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                   .clip(-self.csi_clip, self.csi_clip) +
        #                   rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                   .clip(-self.csi_clip, self.csi_clip) * 1j
        #                   ) * np.array((1 / 5) ** 0.5)

        """curriculum learning"""
        self.curr_schedule = self.get_curr_schedule()
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_bs_rl_ary, self.bs_rl_ary)

        self.get_random_from_fixed_array(self.fixed_ur_rl_ary, self.ur_rl_ary)

        # self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)
        # self.ur_bs_ary = np.concatenate([self.ur_bs_ary, ] * self.rep_copy, axis=0)

        self.step_i = 0
        return self.get_state()  # state

    def get_state(self):
        # self.state_dim = (2, users_n + bases_n, relay_n + bases_n)
        state = np.zeros(self.state_dim, dtype=np.float32)

        # G: bs_rl
        self.bs_rl = self.bs_rl_ary[self.step_i, : self.relay_n]
        state[0, : self.bases_n, : self.relay_n] = self.bs_rl.real
        state[1, : self.bases_n, : self.relay_n] = self.bs_rl.imag

        # H_irs: ur_rl
        self.ur_rl = self.ur_rl_ary[self.step_i, : self.relay_n]
        state[0, self.bases_n :, : self.relay_n] = self.ur_rl.real
        state[1, self.bases_n :, : self.relay_n] = self.ur_rl.imag
        self.ur_rl = self.ur_rl.transpose()

        # # H_dk: ur_bs
        # self.ur_bs = self.ur_bs_ary[self.step_i, :self.relay_n]  # sth wrong
        # state[0, self.bases_n:, self.relay_n:] = self.ur_bs.real
        # state[1, self.bases_n:, self.relay_n:] = self.ur_bs.imag
        # self.ur_bs = self.ur_bs.transpose()
        return state

    def get_reward(self, action):
        phi = action[: self.relay_n] + action[self.relay_n :] * 1j

        temp_list = []
        j_opt = 0
        for i in range(self.users_n):
            h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
            # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)

            variable = (self.bs_rl * phi).dot(h_irs_i)  # + h_dk_i
            variable = variable.reshape(self.bases_n, 1)
            variable_h = variable.transpose().conjugate()

            temp_list.append((variable, variable_h))

            j_opt += self.power * (variable.dot(variable_h))
        j_opt += np.eye(self.bases_n)

        reward = 0
        for i in range(self.users_n):
            variable, variable_h = temp_list[i]

            e_mmse = 1 - self.power * variable_h.dot(np.linalg.inv(j_opt)).dot(variable)
            reward += float(np.real(np.log2(e_mmse) * -1))
        return reward - self.reward_offset

    def get_iter_action(self, iter_n=32):
        random_phi = np.random.normal(0, 1, size=(self.relay_n,)).clip(-1, 1)
        phi_ite_mat = np.diag(np.exp(1j * np.pi * random_phi))
        for k in range(iter_n):
            j_random = 0
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i)  # + h_dk_i
                """------------------------------------------------------------"""
                # self.var_i_mid_list_phi.append(var_i_mid_list)
                """------------------------------------------------------------"""
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                j_random += self.power * (var_i_mid_ite.dot(var_i_mid_ite_h))
            j_random += np.eye(self.bases_n)

            a_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            b_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            c_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            d_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i)  # + h_dk_i
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                """------------------------------------------------------------"""
                # h_irs_i = self.h_irs[:, i].reshape(self.irs_num, 1)
                # var_i_mid_ite = self.var_i_mid_list_phi[i].reshape(self.base_num, 1)
                # var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()         # 用列表加载需要每次清空列表
                """------------------------------------------------------------"""
                w_mat = (np.sqrt(self.power) * np.linalg.inv(j_random)).dot(
                    var_i_mid_ite
                )
                w_mat_h = w_mat.transpose().conjugate()
                e_i = np.real(
                    1
                    - self.power
                    * (var_i_mid_ite_h.dot(np.linalg.inv(j_random)).dot(var_i_mid_ite))
                )
                a_mat += (1 / e_i) * self.bs_rl.transpose().conjugate().dot(w_mat).dot(
                    w_mat_h
                ).dot(self.bs_rl)
                h_irs_h = h_irs_i.transpose().conjugate()
                b_mat += self.power * h_irs_i.dot(h_irs_h)
                d_mat += (1 / e_i) * (
                    np.sqrt(self.power) * h_irs_i.dot(w_mat_h).dot(self.bs_rl)
                )
                # for m in range(self.users_n):
                #     h_irs_m = self.ur_rl[:, m].reshape(self.relay_n, 1)
                #     h_dk_m = self.ur_bs[:, m].reshape(self.bases_n, 1)
                #     h_dk_m_h = h_dk_m.transpose().conjugate()
                #     c_mat += self.power * (1 / e_i) * h_irs_m.dot(h_dk_m_h).dot(w_mat).dot(w_mat_h).dot(self.bs_rl)
            psi = a_mat * b_mat.transpose()
            v = np.diag(c_mat - d_mat)
            eig_val, eig_vct = np.linalg.eig(psi)
            lamba_max = np.max(np.real(eig_val))
            phi_ite = np.diagonal(phi_ite_mat)  # get the eye element
            last_rt = 0
            for ite in range(300):
                phi_ite = np.exp(
                    1j
                    * np.angle(
                        (lamba_max * (np.eye(self.relay_n)) - psi).dot(phi_ite)
                        - v.conjugate()
                    )
                )
                rt = np.real(
                    phi_ite.transpose().conjugate().dot(psi).dot(phi_ite)
                ) + 2 * np.real(phi_ite.transpose().conjugate().dot(v.conjugate()))
                if ite >= 20 and (
                    abs(rt - last_rt) / abs(rt + 0.00000000001) <= 0.0001
                ):
                    phi_ite_mat = np.diag(phi_ite)
                    break
                last_rt = rt
        action = phi_ite_mat.sum(axis=0)
        action = np.hstack((action.real, action.imag))
        return action


class DownLinkEnv5(DownLinkEnv1):
    def __init__(
        self,
        bases_n=2,
        users_n=2,
        relay_n=12,
        power=0.2,
        csi_clip=3.0,
        env_cwd=".",
        curr_delta=0.00,
        curr_schedule_init=0.005,
    ):
        """
        :param bases_n: antennas number of BS (BaseStation)
        :param users_n: user number
        :param relay_n: antennas number of ReLay IRS (Intelligent Reflecting Surfaces)
        :param power: the power of BaseStation. `self.get_action_power()`
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: CWD (current working directory) path for curriculum learning
        :param curr_delta: the delta of schedule value (0.0 ~ 1.0) for curriculum learning
        :param curr_schedule_init: the initialization value of schedule value (0.0 ~ 1.0)
        """
        super(DownLinkEnv1).__init__()
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.relay_n = relay_n  # rl
        self.power = power
        self.csi_clip = csi_clip

        print(f"users_n {users_n}    bases_n {bases_n}    relay_n {relay_n}")
        assert users_n == bases_n
        if users_n == 2:
            self.reward_offset = {
                12: (3.4 + 6.7) / 2,  # user=2, bs=2, relay=12
            }[relay_n]
        elif users_n == 3:
            self.reward_offset = {
                8: (07.1 + 4.5) / 2,  # user=3, bs=3, relay=8
                12: (09.8 + 5.4) / 2,  # user=3, bs=3, relay=12
                16: (11.7 + 6.0) / 2,  # user=3, bs=3, relay=16
            }[relay_n]
        else:
            self.reward_offset = 0.0

        self.env_name = "DownLinkEnv-v5"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (
            2,
            users_n + bases_n,
            relay_n,
        )  # (2, users_n + bases_n, relay_n + bases_n)
        self.action_dim = int(relay_n * 2)
        self.max_step = 2**12
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.bs_rl_ary = self.bs_rl = None
        self.ur_rl_ary = self.ur_rl = None
        # self.ur_bs_ary = self.ur_bs = None

        """curriculum learning"""
        self.curr_delta = curr_delta
        self.curr_target_return = self.reward_offset - self.reward_offset  # 18.4
        self.curr_schedules = (
            [0.001, 0.040, 0.080, 0.100, 0.110]
            + list(np.linspace(0.120, 0.160, 40)[:-1])
            + list(np.linspace(0.160, 0.180, 80)[:-1])
            + list(np.linspace(0.180, 0.200, 20)[:-1])
            + list(np.linspace(0.200, 0.500, 20)[:-1])
            + list(np.linspace(0.500, 1.000, 10)[:-1])
        )

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{curr_schedule_init}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_bs_rl_ary = np_save_dict["fixed_bs_rl"]
            self.fixed_ur_rl_ary = np_save_dict["fixed_rl_ur"]
            # self.fixed_ur_bs_ary = np_save_dict['fixed_ur_bs']

    def save_fixed_data_in_disk(self):
        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        self.fixed_bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        self.fixed_ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * ((1 / 2) ** 0.5)

        # self.fixed_ur_bs_ary = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                         .clip(-self.csi_clip, self.csi_clip) +
        #                         rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                         .clip(-self.csi_clip, self.csi_clip) * 1j
        #                         ) * np.array((1 / 5) ** 0.5)

        np_save_dict = {
            "fixed_bs_rl": self.fixed_bs_rl_ary,
            "fixed_rl_ur": self.fixed_ur_rl_ary,
            # 'fixed_ur_bs': self.fixed_ur_bs_ary,
        }
        np.savez(self.npy_data_path, **np_save_dict)
        print(f"DownLink-v2: {self.npy_data_path}")

    def reset(self):
        # G: bs_rl
        self.bs_rl_ary = (
            rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.bases_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # H_irs: ur_rl
        self.ur_rl_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.relay_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # # H_dk: ur_bs
        # self.ur_bs_ary = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                   .clip(-self.csi_clip, self.csi_clip) +
        #                   rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #                   .clip(-self.csi_clip, self.csi_clip) * 1j
        #                   ) * np.array((1 / 5) ** 0.5)

        """curriculum learning"""
        self.curr_schedule = self.get_curr_schedule()
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_bs_rl_ary, self.bs_rl_ary)

        self.get_random_from_fixed_array(self.fixed_ur_rl_ary, self.ur_rl_ary)

        # self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)
        # self.ur_bs_ary = np.concatenate([self.ur_bs_ary, ] * self.rep_copy, axis=0)

        self.step_i = 0
        return self.get_state()  # state

    def get_state(self):
        # self.state_dim = (2, users_n + bases_n, relay_n + bases_n)
        state = np.zeros(self.state_dim, dtype=np.float32)

        # G: bs_rl
        self.bs_rl = self.bs_rl_ary[self.step_i, : self.relay_n]
        state[0, : self.bases_n, : self.relay_n] = self.bs_rl.real
        state[1, : self.bases_n, : self.relay_n] = self.bs_rl.imag

        # H_irs: ur_rl
        self.ur_rl = self.ur_rl_ary[self.step_i, : self.relay_n]
        state[0, self.bases_n :, : self.relay_n] = self.ur_rl.real
        state[1, self.bases_n :, : self.relay_n] = self.ur_rl.imag
        self.ur_rl = self.ur_rl.transpose()

        # # H_dk: ur_bs
        # self.ur_bs = self.ur_bs_ary[self.step_i, self.relay_n:]
        # state[0, self.bases_n:, self.relay_n:] = self.ur_bs.real
        # state[1, self.bases_n:, self.relay_n:] = self.ur_bs.imag
        # self.ur_bs = self.ur_bs.transpose()
        return state

    def get_reward(self, action):
        phi = action[: self.relay_n] + action[self.relay_n :] * 1j

        temp_list = []
        j_opt = 0
        for i in range(self.users_n):
            h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
            # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)

            variable = (self.bs_rl * phi).dot(h_irs_i)  # + h_dk_i
            variable = variable.reshape(self.bases_n, 1)
            variable_h = variable.transpose().conjugate()

            temp_list.append((variable, variable_h))

            j_opt += self.power * (variable.dot(variable_h))
        j_opt += np.eye(self.bases_n)

        reward = 0
        for i in range(self.users_n):
            variable, variable_h = temp_list[i]

            e_mmse = 1 - self.power * variable_h.dot(np.linalg.inv(j_opt)).dot(variable)
            reward += float(np.real(np.log2(e_mmse) * -1))
        return reward - self.reward_offset

    def get_iter_action(self, iter_n=32):
        random_phi = np.random.normal(0, 1, size=(self.relay_n,)).clip(-1, 1)
        phi_ite_mat = np.diag(np.exp(1j * np.pi * random_phi))
        for k in range(iter_n):
            j_random = 0
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i)  # + h_dk_i
                """------------------------------------------------------------"""
                # self.var_i_mid_list_phi.append(var_i_mid_list)
                """------------------------------------------------------------"""
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                j_random += self.power * (var_i_mid_ite.dot(var_i_mid_ite_h))
            j_random += np.eye(self.bases_n)

            a_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            b_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            c_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            d_mat = np.zeros((self.relay_n, self.relay_n), dtype=complex)
            for i in range(self.users_n):
                """------------------------------------------------------------"""
                h_irs_i = self.ur_rl[:, i].reshape(self.relay_n, 1)
                # h_dk_i = self.ur_bs[:, i].reshape(self.bases_n, 1)
                var_i_mid_list = self.bs_rl.dot(phi_ite_mat).dot(h_irs_i)  # + h_dk_i
                var_i_mid_ite = var_i_mid_list.reshape(self.bases_n, 1)
                var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()
                """------------------------------------------------------------"""
                # h_irs_i = self.h_irs[:, i].reshape(self.irs_num, 1)
                # var_i_mid_ite = self.var_i_mid_list_phi[i].reshape(self.base_num, 1)
                # var_i_mid_ite_h = var_i_mid_ite.transpose().conjugate()         # 用列表加载需要每次清空列表
                """------------------------------------------------------------"""
                w_mat = (np.sqrt(self.power) * np.linalg.inv(j_random)).dot(
                    var_i_mid_ite
                )
                w_mat_h = w_mat.transpose().conjugate()
                e_i = np.real(
                    1
                    - self.power
                    * (var_i_mid_ite_h.dot(np.linalg.inv(j_random)).dot(var_i_mid_ite))
                )
                a_mat += (1 / e_i) * self.bs_rl.transpose().conjugate().dot(w_mat).dot(
                    w_mat_h
                ).dot(self.bs_rl)
                h_irs_h = h_irs_i.transpose().conjugate()
                b_mat += self.power * h_irs_i.dot(h_irs_h)
                d_mat += (1 / e_i) * (
                    np.sqrt(self.power) * h_irs_i.dot(w_mat_h).dot(self.bs_rl)
                )
                # for m in range(self.users_n):
                #     h_irs_m = self.ur_rl[:, m].reshape(self.relay_n, 1)
                #     h_dk_m = self.ur_bs[:, m].reshape(self.bases_n, 1)
                #     h_dk_m_h = h_dk_m.transpose().conjugate()
                #     c_mat += self.power * (1 / e_i) * h_irs_m.dot(h_dk_m_h).dot(w_mat).dot(w_mat_h).dot(self.bs_rl)
            psi = a_mat * b_mat.transpose()
            v = np.diag(c_mat - d_mat)
            eig_val, eig_vct = np.linalg.eig(psi)
            lamba_max = np.max(np.real(eig_val))
            phi_ite = np.diagonal(phi_ite_mat)  # get the eye element
            last_rt = 0
            for ite in range(300):
                phi_ite = np.exp(
                    1j
                    * np.angle(
                        (lamba_max * (np.eye(self.relay_n)) - psi).dot(phi_ite)
                        - v.conjugate()
                    )
                )
                rt = np.real(
                    phi_ite.transpose().conjugate().dot(psi).dot(phi_ite)
                ) + 2 * np.real(phi_ite.transpose().conjugate().dot(v.conjugate()))
                if ite >= 20 and (
                    abs(rt - last_rt) / abs(rt + 0.00000000001) <= 0.0001
                ):
                    phi_ite_mat = np.diag(phi_ite)
                    break
                last_rt = rt
        action = phi_ite_mat.sum(axis=0)
        action = np.hstack((action.real, action.imag))
        return action

    def curriculum_learning_for_evaluator(self, r_avg):
        """Call this function in `Class Evaluator.evaluate_and_save()`
        if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
            self.eval_env.curriculum_learning_for_evaluator(r_avg)
        """
        if r_avg < self.curr_target_return or self.curr_schedule == 1:
            return

        """update curriculum learning tau in disk"""
        # self.curr_schedule = min(1.0, self.curr_schedule + self.curr_delta)  # write before `self.reset()`
        self.curr_schedule = min(
            1.0, self.curr_schedule + self.curr_delta
        )  # write before `self.reset()`

        if self.curr_schedules:
            self.curr_schedule = self.curr_schedules[0]
            del self.curr_schedules[0]
        else:
            self.curr_schedule = 1.0

        with open(self.curr_txt_path, "w+") as f:
            f.write(f"{self.curr_schedule}\n")

        print(
            f"\n| DownLinkEnv: {self.curr_schedule:8.3f}    "
            f"currR {self.curr_target_return:8.3f}    avgR {r_avg:8.3f}"
        )


class DownLinkEnv21(DownLinkEnv1):  # [ElegantRL.2021.12.12]
    def __init__(
        self,
        bases_n=4,
        users_n=8,
        power=1.0,
        sigma=1.0,
        csi_noise_var=0.0,
        csi_clip=3.0,
        env_cwd=".",
        curr_schedules=None,
    ):
        """
        :param bases_n: antennas number of BaseStation
        :param users_n: user number
        :param power: the power of BaseStation. `self.get_action_power()`
        :param sigma: SNR=MaxPower/sigma**2, suggest tp keep power=1.0
        :param csi_noise_var: the noise var of CSI (Channel State Information)
        :param csi_clip: clip the CSI which obeys normal distribution
        :param env_cwd: [str] CWD (current working directory) path for curriculum learning
        :param curr_schedules: [list] the list of schedule value (0.0 ~ 1.0]
        """
        super(DownLinkEnv1).__init__()
        self.bases_n = bases_n  # bs
        self.users_n = users_n  # ur
        self.power = power
        assert self.power == 1.0
        self.sigma = sigma
        self.csi_clip = csi_clip
        self.csi_noise_var = csi_noise_var  # 0.0, 0.1
        assert self.csi_noise_var == 0

        self.env_name = "DownLinkEnv-v2"
        self.env_num = 1
        self.env_cwd = env_cwd
        self.state_dim = (2, users_n, bases_n)
        self.action_dim = users_n
        self.max_step = 2**14
        self.if_discrete = False
        self.target_return = +np.inf * self.max_step

        self.step_i = None
        self.ur_bs_ary = None  # Hall
        self.ur_bs_ary_noisy = None  # noisy Hall

        """curriculum learning"""
        # csi_noise_var=0.0, power = 1, sigma=1
        self.reward_offset = np.array(
            [
                [1.0, 2.424],
                [2.0, 2.906],
                [3.0, 2.794],
                [4.0, 2.339],
                [5.0, 2.385],
                [6.0, 2.547],
                [7.0, 2.652],
                [8.0, 2.713],
            ]
        )[users_n - 1, 1]
        self.curr_target_return = self.reward_offset - self.reward_offset
        self.curr_schedules = (
            (
                [
                    0.01,
                    0.05,
                    0.10,
                ]
                + list(np.linspace(0.20, 1.00, 20 - 1)[:-1])
            )
            if curr_schedules is None
            else curr_schedules
        )

        self.curr_txt_path = f"{self.env_cwd}/tau_of_curriculum.txt"
        if os.path.isfile(self.curr_txt_path):
            self.curr_schedule = self.get_curr_schedule()
        else:
            if not os.path.isdir(self.env_cwd):
                os.mkdir(self.env_cwd)
            with open(self.curr_txt_path, "w+") as f:
                f.write(f"{self.curr_schedules[0]}\n")

        self.npy_data_path = f"{self.env_cwd}/fixed_data.npz"
        if os.path.isfile(self.npy_data_path):
            np_save_dict = np.load(self.npy_data_path)

            self.fixed_ur_bs_ary = np_save_dict["fixed_ur_bs"]
        else:
            print(f"DownLinkEnv: FileNotFound {self.npy_data_path}")

    def reset(self):
        # Hall: ur_bs
        self.ur_bs_ary = (
            rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            + rd.randn(self.max_step + 1, self.users_n, self.bases_n)
            .astype(np.float32)
            .clip(-self.csi_clip, self.csi_clip)
            * 1j
        ) * np.array((1 / 2) ** 0.5)

        # Hall:
        # noise = (rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #          .clip(-self.csi_clip, self.csi_clip) +
        #          rd.randn(self.max_step + 1, self.users_n, self.bases_n).astype(np.float32)
        #          .clip(-self.csi_clip, self.csi_clip) * 1j
        #          ) * np.array((self.csi_noise_var / 2) ** 0.5)
        # self.ur_bs_ary_noisy = self.ur_bs_ary + noise
        # assert self.csi_noise_var == 0
        self.ur_bs_ary_noisy = self.ur_bs_ary

        """curriculum learning"""
        self.curr_schedule = self.get_curr_schedule()
        assert 0.0 <= self.curr_schedule <= 1.0
        self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary)
        # self.get_random_from_fixed_array(self.fixed_ur_bs_ary, self.ur_bs_ary_noisy)
        # assert self.csi_noise_var == 0

        self.step_i = 0
        return self.get_state()  # state

    def step(self, action):  # todo
        # h =  # todo
        reward = self.get_reward(action)

        self.step_i += 1  # write before `self.get_state()`
        state = self.get_state()

        done = self.step_i == self.max_step
        return state, reward, done, None

    def get_state(self):
        hall_noisy = self.ur_bs_ary_noisy[self.step_i]
        return np.stack((hall_noisy.real, hall_noisy.imag))

    def get_reward(self, action):  # get_sum_rate we need a sigma here(maybe).
        h = self.ur_bs_ary[self.step_i]

        action = action.reshape((2, self.bases_n, self.users_n))
        w = action[0] + action[1] * 1j
        return get_sum_rate_miso(h, w, self.sigma) - self.reward_offset


def get_sum_rate_mimo(h, w, sigma=1.0) -> float:
    """
    bases_n: antennas number of base station
    users_n: number of user
    u_ant_n: antennas number of each user

    :param h: hall, (state), h.shape == (users_n, u_ant_n, bases_n)
    :param w: weight, (action), w.shape == (bases_n, users_n, u_ant_n)
    :param sigma: SNR = Power / sigma**2
    :return: isinstance(rates.sum(), float), rates.shape = (users_n,)
    """
    bases_n, users_n, u_ant_n = w.shape
    w_tilde = np.empty((users_n, bases_n, u_ant_n * (users_n - 1)), dtype=w.dtype)
    for n in range(users_n):
        j1 = 0
        for j0 in range(users_n):
            if j0 != n:
                w_tilde[n, :, j1 : j1 + u_ant_n] = w[:, j0]
                j1 += u_ant_n
    # assert w_tilde.shape == (users_n, bases_n, u_ant_n * (users_n - 1))

    u_ant_eye = np.eye(u_ant_n)[np.newaxis]

    h_c = h.conj().transpose((0, 2, 1))
    w_tilde_c = w_tilde.conj().transpose((0, 2, 1))
    sigma_k = u_ant_eye * sigma**2 + h @ w_tilde @ w_tilde_c @ h_c
    # assert sigma_k.shape == (users_n, u_ant_n, u_ant_n)

    w1 = w.transpose((1, 0, 2))
    w1c = w1.conj().transpose((0, 2, 1))
    rate0 = u_ant_eye + np.linalg.inv(sigma_k) @ h @ w1 @ w1c @ h_c
    # assert rate0.shape == (users_n, u_ant_n, u_ant_n)
    # assert np.linalg.det(rate0).shape == (users_n, )
    return np.abs(np.log2(np.linalg.det(rate0))).sum()


def get_sum_rate_miso_torch(h, w, sigma=1.0) -> torch.tensor:
    """
    :param h: hall, (state), h.shape == (users_n, bases_n)
    :param w: weight, (action), w.shape == (bases_n, users_n)
    :param sigma: SNR = Power / sigma**2
    :return: isinstance(rates.sum(), float), rates.shape = (users_n,)
    """
    channel_gains = torch.pow(torch.abs(torch.matmul(h, w)), 2)
    # assert channel_gains.shape == (users_n, users_n)
    signal_gains = torch.diag(channel_gains)
    # assert signal_gains.shape == (users_n, )
    interference_gains = channel_gains.sum(dim=1) - signal_gains
    rates = torch.log2(1 + signal_gains / (interference_gains + sigma**2))
    return rates.sum()


def get_sum_rate_miso_torch_vec(h, w, sigma=1.0) -> torch.tensor:
    """
    :param h: hall, (batch_size, *state), h.shape == (batch_size, users_n, bases_n)
    :param w: weight, (batch_size, *action), w.shape == (batch_size, bases_n, users_n)
    :param sigma: SNR = Power / sigma**2
    :return: isinstance(rates.sum(), float), rates.shape = (users_n,)
    """
    channel_gains = torch.pow(torch.abs(torch.matmul(h, w)), 2)
    signal_gains = torch.diagonal(channel_gains, offset=0, dim1=-2, dim2=-1)
    interference_gains = channel_gains.sum(dim=2) - signal_gains
    rates = torch.log2(1 + signal_gains / (interference_gains + sigma**2))
    return rates.sum(dim=1)


def get_min_rate_miso_torch_vec(h, w, sigma=1.0) -> torch.tensor:
    """
    :param h: hall, (batch_size, *state), h.shape == (batch_size, users_n, bases_n)
    :param w: weight, (batch_size, *action), w.shape == (batch_size, bases_n, users_n)
    :param sigma: SNR = Power / sigma**2
    :return: isinstance(rates.sum(), float), rates.shape = (users_n,)
    """
    channel_gains = torch.pow(torch.abs(torch.matmul(h, w)), 2)
    signal_gains = torch.diagonal(channel_gains, offset=0, dim1=-2, dim2=-1)
    interference_gains = channel_gains.sum(dim=2) - signal_gains
    rates = torch.log2(1 + signal_gains / (interference_gains + sigma**2))
    return torch.min(rates, dim=1)[0]


def func_slnr_max(h, eta, user_k, antennas_n):
    w_slnr_max_list = []

    for k in range(user_k):
        effective_channel = h.conj().T  # h'

        projected_channel = np.eye(antennas_n) / eta + np.dot(effective_channel, h)
        projected_channel = np.linalg.solve(projected_channel, effective_channel[:, k])

        w_slnr_max = projected_channel / np.linalg.norm(projected_channel)
        w_slnr_max_list.append(w_slnr_max)

    return np.stack(w_slnr_max_list).T  # (antennas_n, user_k)


def func_mmse(h_noisy, bases_n, users_n, power, csi_noise_var):
    # assert h_noisy.shape == (users_n, bases_n)

    h_tilde = h_noisy * (1 / (csi_noise_var + 1))

    eta = power * users_n
    w_mmse_norm = func_slnr_max(h_tilde, eta, users_n, bases_n)
    power_mmse = np.ones((1, users_n)) * (power / users_n)

    w_mmse = np.power(power_mmse, 0.5) * np.ones((bases_n, 1))
    w_mmse = w_mmse * w_mmse_norm
    return w_mmse  # action


def func_mmse_vec(h_noisy, bases_n, users_n, power, csi_noise_var):
    # assert h_noisy.shape == (vec_env, users_n, bases_n)

    h_tilde = h_noisy * (1 / (csi_noise_var + 1))

    eta = power * users_n
    # w_mmse_norm = func_slnr_max_vec(h_tilde, eta)
    # power_mmse = np.ones((1, users_n)) * (power / users_n)
    #
    # w_mmse = np.power(power_mmse, 0.5) * np.ones((bases_n, 1))
    # w_mmse = w_mmse * w_mmse_norm
    return func_slnr_max_vec(h_tilde, eta)  # action


def func_slnr_max_vec(h_vec, eta_vec: float):
    batch_size, user_k, bases_n = h_vec.shape

    w_mmse_vec = torch.empty(
        (batch_size, bases_n, user_k), dtype=h_vec.dtype, device=h_vec.device
    )
    eye_div_eta = torch.eye(bases_n, dtype=h_vec.dtype, device=h_vec.device) / eta_vec

    effective_channel = h_vec.conj().transpose(2, 1)
    projected_channel_vec = eye_div_eta.unsqueeze(0) + torch.matmul(
        effective_channel, h_vec
    )
    # assert projected_channel_vec.shape(batch_size, user_k, user_k)
    for bs in range(batch_size):
        for k in range(user_k):
            projected_channel = torch.linalg.solve(
                projected_channel_vec[bs], effective_channel[bs, :, k]
            )
            w_mmse_vec[bs, :, k] = projected_channel

    w_mmse_vec = w_mmse_vec / (torch.norm(w_mmse_vec, dim=1, keepdim=True) * user_k)
    return w_mmse_vec  # (batch_size, bases_n, user_k)


"""check"""


def check__get_sum_rate():
    bases_n = 5
    users_n = 4
    u_ant_n = 1
    sigma = 1.234

    if_mimo = 1
    h0 = rd.rand(users_n, u_ant_n, bases_n) + rd.rand(users_n, u_ant_n, bases_n) * 1j
    w0 = rd.rand(bases_n, users_n, u_ant_n) + rd.rand(bases_n, users_n, u_ant_n) * 1j
    if if_mimo:
        h1 = h0.reshape((users_n, u_ant_n, bases_n))
        w1 = w0.reshape((bases_n, users_n, u_ant_n))

        r1 = get_sum_rate_mimo(h=h1, w=w1, sigma=sigma)
        print(f"| get_mimo_sum_rate() {r1:8.4f}")

    if_miso = 1
    if if_miso:
        assert u_ant_n == 1
        h1 = h0.reshape((users_n, bases_n))
        w1 = w0.reshape((bases_n, users_n))
        r1 = get_sum_rate_miso(h=h1, w=w1, sigma=sigma)
        print(f"| get_miso_sum_rate() {r1:8.4f}")

    if_miso_torch = 1
    if if_miso_torch:
        h3 = torch.as_tensor(h0).reshape((users_n, bases_n))
        w3 = torch.as_tensor(w0).reshape((bases_n, users_n))
        r3 = get_sum_rate_miso_torch(h3, w3, sigma=sigma)
        print(f"| get_sum_rate_miso_torch() {r3.item():8.4f}")

    if_miso_torch_vec = 1
    if if_miso_torch_vec:
        batch_size = 2
        batch_ones = torch.ones(batch_size, 1, 1)
        batch_ones[1] = 0.5

        h4 = torch.as_tensor(h0).reshape((1, users_n, bases_n)) * batch_ones
        w4 = torch.as_tensor(w0).reshape((1, bases_n, users_n)) * batch_ones
        r4 = get_sum_rate_miso_torch_vec(h4, w4, sigma=sigma)
        print(f"| get_sum_rate_miso_torch() {r4[0].item():8.4f}")
        print(f"| get_sum_rate_miso_torch() {r4[1].item():8.4f}")


def check_sum_rate():
    power = 1.0
    snr_db = 15
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(power / snr)  # 0.17782794100389226
    # sigma = 1

    ary_h = [
        (
            -0.4790 + 0.8240j,
            -0.6171 + 0.4939j,
            0.1445 - 0.8305j,
            0.7301 + 0.0938j,
            0.4158 + 0.2001j,
            -0.0896 + 0.7613j,
        ),
        (
            0.3909 + 0.5425j,
            -0.1947 - 0.6609j,
            0.6870 - 0.2159j,
            -0.2455 + 0.5149j,
            0.1126 - 0.7439j,
            0.0216 - 0.4870j,
        ),
        (
            -1.3133 - 0.9003j,
            -0.8099 - 0.3565j,
            -1.3273 - 0.3383j,
            0.2799 + 0.7595j,
            -0.0928 - 0.2466j,
            0.0259 - 0.1011j,
        ),
        (
            -0.7355 + 1.6057j,
            1.3985 - 0.0684j,
            -0.1798 + 0.2252j,
            -0.2462 + 0.4109j,
            0.3284 + 0.2269j,
            -0.3877 - 0.7796j,
        ),
    ]

    ary_w = [
        (
            0.1409 + 0.0034j,
            -0.0118 + 0.1800j,
            0.0944 + 0.2568j,
            0.1795 - 0.0257j,
            0.1740 - 0.1521j,
            -0.1698 - 0.1031j,
        ),
        (
            0.1889 - 0.0598j,
            -0.1893 + 0.1019j,
            -0.1592 + 0.2121j,
            0.0923 + 0.1326j,
            -0.2310 - 0.2276j,
            0.0577 + 0.2034j,
        ),
        (
            -0.1028 + 0.1879j,
            -0.0319 - 0.1323j,
            0.2026 - 0.2438j,
            0.2002 - 0.0749j,
            -0.0337 - 0.1094j,
            -0.0550 + 0.0940j,
        ),
        (
            0.1182 + 0.0000j,
            0.2687 + 0.0000j,
            0.1688 + 0.0000j,
            0.1466 + 0.0000j,
            0.0911 + 0.0000j,
            0.1876 + 0.0000j,
        ),
    ]

    ten_h = torch.as_tensor(ary_h, dtype=torch.complex128)
    ten_w = torch.as_tensor(ary_w, dtype=torch.complex128)
    ten_h = torch.transpose(ten_h.conj(), 1, 0)

    r = get_min_rate_miso_torch_vec(ten_h.unsqueeze(0), ten_w.unsqueeze(0), sigma)

    print(r)
    """
    tensor([[1.3194, 1.3159, 1.3171, 1.3166, 1.3152, 1.3155]], dtype=torch.float64)
    tensor([1.3152], dtype=torch.float64)
    """


def get_episode_return_of_mmse(env):
    epi_return = 0.0
    state = env.reset()
    for _ in range(env.max_step):
        # action = rd.randn(env.action_dim)
        action = env.get_action_mmse(state)

        state, reward, done, info_dict = env.step(action)

        epi_return += reward
    epi_return /= env.max_step
    return epi_return


def check__mmse_on_env():
    import time

    bases_n = 4
    users_n = 8
    csi_noise_var = 0.0

    env = DownLinkEnv1(
        bases_n=bases_n, users_n=users_n, power=1.0, csi_noise_var=csi_noise_var
    )
    env.curr_schedule = 1.0
    env.max_step = 2**10

    timer = time.time()

    if_change_power = False
    if if_change_power:
        power_db_ary = np.arange(-10, 30 + 5, 5)  # SNR (dB)
        power_ary = 10 ** (power_db_ary / 10)
        show_ary = []
        for j, power in enumerate(power_ary):
            # env = DownLinkEnv0(bases_n=bases_n, users_n=users_n, power=power, csi_noise_var=csi_noise_var)
            env.power = power

            episode_return = get_episode_return_of_mmse(env)
            show_ary.append((power_db_ary[j], episode_return))

        show_ary = np.array(show_ary).round(3)
        print(show_ary)
        print(f"UsedTime: {time.time() - timer:8.3f}")

        """
        users_n = 4
        [[-10.      0.42 ]
         [ -5.      0.977]
         [  0.      1.98 ]
         [  5.      3.399]
         [ 10.      5.099]
         [ 15.      6.799]
         [ 20.      8.021]
         [ 25.      8.558]
         [ 30.      8.776]]
        UsedTime:    2.116
        
        users_n = 6
        [[-10.      0.387]
         [ -5.      0.932]
         [  0.      2.073]
         [  5.      3.793]
         [ 10.      5.615]
         [ 15.      6.708]
         [ 20.      7.229]
         [ 25.      7.39 ]
         [ 30.      7.454]]
        UsedTime:    2.825    
        
        users_n = 8
        [[-10.      0.383]
         [ -5.      0.966]
         [  0.      2.2  ]
         [  5.      3.916]
         [ 10.      5.409]
         [ 15.      6.185]
         [ 20.      6.447]
         [ 25.      6.558]
         [ 30.      6.605]]
        UsedTime:    3.823
        """

    if_change_users_num = True
    if if_change_users_num:
        user_ary = [i for i in range(1, 8 + 1)]
        show_ary = []
        for j, users_n in enumerate(user_ary):
            env = DownLinkEnv1(
                bases_n=bases_n, users_n=users_n, power=1.0, csi_noise_var=csi_noise_var
            )
            env.save_fixed_data_in_disk()
            env = DownLinkEnv1(
                bases_n=bases_n, users_n=users_n, power=1.0, csi_noise_var=csi_noise_var
            )
            env.power = 1.0
            env.users_n = users_n

            episode_return = get_episode_return_of_mmse(env)
            show_ary.append((user_ary[j], episode_return))

        show_ary = np.array(show_ary).round(3)
        print(repr(show_ary))
        print(f"UsedTime: {time.time() - timer:8.3f}")

        """
        sigma=1
        power=1
        csi_noise_var = 0
        [[1.    2.203]
         [2.    2.592]
         [3.    2.442]
         [4.    2.096]
         [5.    2.12 ]
         [6.    2.248]
         [7.    2.333]
         [8.    2.41 ]]
        
        sigma=1
        power=1
        csi_noise_var = 0.1
        [[1.    2.106]
         [2.    2.461]
         [3.    2.321]
         [4.    2.017]
         [5.    1.964]
         [6.    2.035]
         [7.    2.137]
         [8.    2.197]]
        UsedTime:    5.436
        """


def check__mmse_on_random_data():
    import time

    bases_n = 4  # antennas number of BaseStation
    users_n = 8  # user number
    sim_times = 2**10  # simulate times
    csi_noise_var = 0.1  # channel state information
    sigma = 1.0

    power_db_ary = np.arange(-10, 30 + 5, 5)  # SNR (dB)
    power_ary = 10 ** (power_db_ary / 10)

    timer = time.time()
    show_ary = []
    for j, power in enumerate(power_ary):
        """generate state"""
        hall_ary = (
            rd.randn(sim_times, users_n, bases_n)
            + rd.randn(sim_times, users_n, bases_n) * 1j
        )
        hall_ary *= (1 / 2) ** 0.5
        hall_noise_ary = (
            rd.randn(sim_times, users_n, bases_n)
            + rd.randn(sim_times, users_n, bases_n) * 1j
        )
        hall_noise_ary *= (csi_noise_var / 2) ** 0.5

        sum_rate_temp = 0
        for t in range(sim_times):
            hall = hall_ary[t]
            hall_noisy = hall + hall_noise_ary[t]

            w_mmse = func_mmse(hall_noisy, bases_n, users_n, power, csi_noise_var)
            # '''check the power of w'''
            # power_temp = (w_mmse.real ** 2).sum() + (w_mmse.imag ** 2).sum()
            # assert power_temp < power + 0.0001

            sum_rate_temp += get_sum_rate_miso(hall, w_mmse, sigma)

        show_ary.append((power_db_ary[j], sum_rate_temp / sim_times))

    show_ary = np.array(show_ary).round(3)
    print(show_ary)
    print(f"UsedTime: {time.time() - timer:8.3f}")
    """
    [[-10.     -5.      0.      5.     10.     15.     20.     25.     30.   ]
     [  0.383   0.977   2.207   3.94    5.398   6.179   6.478   6.584   6.612]]
    UsedTime:   12.267
    sim_times = 2 ** 12  # simulate times
    """


def get_max_power_action(action):
    # action2 = action.reshape((2, -1))
    # action_normal = np.sqrt(np.power(action2, 2).sum(axis=0, keepdims=True))
    # return (action2 / action_normal).reshape(-1)  # action
    return action / np.linalg.norm(action)


def check__down_link_relay():
    simulate_times = 2**0

    env = DownLinkEnv3()
    if os.path.exists(env.npy_data_path):
        os.remove(env.npy_data_path)
    env.curr_schedule = 1.0

    from tqdm import trange

    episode_returns = []
    for _ in trange(simulate_times):
        env.reset()

        episode_return = 0.0
        for _ in range(env.max_step):
            action = rd.randn(env.action_dim).astype(np.float32).clip(-1, 1)
            # action = np.ones(env.action_dim, dtype=np.float32)
            action = get_max_power_action(action)

            state, reward, done, _ = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return / env.max_step)
    print("Reward(Random)", np.mean(episode_returns), np.std(episode_returns))

    from tqdm import trange

    episode_returns = []
    for simulate_time in range(simulate_times):
        env.reset()

        episode_return = 0.0
        time.sleep(1)
        for _ in trange(env.max_step):
            # action = rd.randn(env.action_dim).astype(np.float32).clip(-1, 1)
            # action = np.ones(env.action_dim, dtype=np.float32)
            action = env.get_iter_action(iter_n=32)
            action = get_max_power_action(action)

            state, reward, done, _ = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return / env.max_step)
        print(
            f"| simulate_time {simulate_time}    episode_return {episode_returns[-1]:8.3f}"
        )

    print("Reward(Random)", np.mean(episode_returns), np.std(episode_returns))


if __name__ == "__main__":
    check__get_sum_rate()
    # check__mmse_on_env()
    # check__down_link_relay()
    # run_mmse_on_random_data()
    pass
