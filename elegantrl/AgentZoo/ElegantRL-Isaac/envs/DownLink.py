import torch
import numpy as np
import numpy.random as rd

"""[ElegantRL.2021.11.04](https://github.com/AI4Finance-Foundation/ElegantRL)"""


class DownLinkEnv:  # stable 2021-11-04
    def __init__(self, bs_n=4, ur_n=8, power=1.0,
                 csi_noise_var=0.1, csi_clip=3.0):
        """
        :param bs_n: antennas number of BaseStation
        :param ur_n: user number
        :param power: the power of BaseStation. `self.get_action_power()`
        :param csi_noise_var: the noise var of Channel State Information
        """
        self.bs_n = bs_n
        self.ur_n = ur_n
        self.power = power
        self.csi_clip = csi_clip
        self.csi_noise_var = csi_noise_var

        self.env_name = 'DownLinkEnv-v0'
        self.env_num = 1
        self.state_dim = (2, ur_n, bs_n)
        self.action_dim = int(np.prod((2, bs_n, ur_n)))
        self.max_step = 2 ** 10
        self.if_discrete = False
        self.target_return = 1943

        self.state = None
        self.step_i = None
        self.hall_ary = None
        self.hall_noisy_ary = None

    def reset(self):
        self.hall_ary = rd.randn(self.max_step + 1, 2, self.ur_n, self.bs_n).clip(-self.csi_clip, self.csi_clip)
        self.hall_ary *= (1 / 2) ** 0.5

        hall_noise_ary = rd.randn(self.max_step + 1, 2, self.ur_n, self.bs_n).clip(-self.csi_clip, self.csi_clip)
        hall_noise_ary *= (self.csi_noise_var / 2) ** 0.5

        self.hall_noisy_ary = self.hall_ary + hall_noise_ary

        self.step_i = 0
        self.state = self.get_state()
        return self.state

    def step(self, action):
        reward = self.get_reward(action)
        self.step_i += 1  # write before `self.get_state()`
        self.state = self.get_state()
        done = self.step_i == self.max_step
        return self.state, reward, done, None

    def get_state(self):
        state = self.hall_noisy_ary[self.step_i]
        return state

    def get_reward(self, action):  # get_sum_rate
        hall = self.hall_ary[self.step_i]
        h = hall[0] + hall[1] * 1j

        action = action.reshape((2, self.bs_n, self.ur_n))
        w = action[0] + action[1] * 1j
        return get_sum_rate(h, w)

    def get_action_norm_power(self, action=None):
        if action is None:
            action = rd.randn(self.action_dim)
        action *= (self.power * (action ** 2).sum()) ** -0.5
        # action /= (self.power * ((action[0] ** 2).sum() + (action[1] ** 2).sum())) ** -0.5
        return action

    def get_action_mmse(self, state):  # not-necessary
        # (state, bs_n, user_n, power, csi_noise_var)
        h_noisy = state[0] + state[1] * 1j
        w_mmse = func_mmse(h_noisy, self.bs_n, self.ur_n, self.power, self.csi_noise_var)
        action = np.stack((w_mmse.real, w_mmse.imag))
        return action


def get_sum_rate(h, w) -> float:
    """
    :param h: hall, (state), h.shape == (user_n, bs_n)
    :param w: weight, (action), w.shape == (bs_n, user_n)
    :return: isinstance(rates.sum(), float), rates.shape = (user_n,)
    """
    channel_gains = np.power(np.abs(np.dot(h, w)), 2)
    signal_gains = np.diag(channel_gains)
    interference_gains = channel_gains.sum(axis=1) - signal_gains
    rates = np.log2(1 + signal_gains / (interference_gains + 1))
    return rates.sum()


def get_sum_rate_gpu(h, w) -> float:
    """
    :param h: hall, (state), h.shape == (n, user_n, bs_n)
    :param w: weight, (action), w.shape == (n, bs_n, user_n)
    :return: isinstance(rates.sum(), float), rates.shape = (user_n,)
    """
    channel_gains = torch.pow(torch.abs(torch.dot(h, w)), 2)
    signal_gains = torch.diag(channel_gains)
    interference_gains = channel_gains.sum(dim=1) - signal_gains
    rates = torch.log2(signal_gains / (interference_gains + 1) + 1)
    return rates.sum()


'''check'''


def func_slnr_max(h, eta, user_k, antennas_n):
    w_slnr_max_list = list()

    for k in range(user_k):
        effective_channel = h.conj().T  # h'

        projected_channel = np.eye(antennas_n) / eta + np.dot(effective_channel, h)
        projected_channel = np.linalg.solve(projected_channel, effective_channel[:, k])

        w_slnr_max = projected_channel / np.linalg.norm(projected_channel)
        w_slnr_max_list.append(w_slnr_max)

    return np.stack(w_slnr_max_list).T  # (antennas_n, user_k)


def func_mmse(h_noisy, bs_n, user_n, power, csi_noise_var):
    # assert h_noisy.shape == (user_n, bs_n)

    h_tilde = (1 / (1 + csi_noise_var)) * h_noisy

    eta = power * user_n
    w_mmse_norm = func_slnr_max(h_tilde, eta, user_n, bs_n)
    power_mmse = np.ones((1, user_n)) * (power / user_n)

    w_mmse = np.power(power_mmse, 0.5) * np.ones((bs_n, 1))
    w_mmse = w_mmse * w_mmse_norm
    return w_mmse  # action


def check__mmse_on_env():
    import time

    bs_n = 4
    user_n = 8
    power = 1.0
    csi_noise_var = 0.1

    env = DownLinkEnv(bs_n, user_n, power, csi_noise_var)
    env.max_step = 2 ** 10

    power_db_ary = np.arange(-10, 30 + 5, 5)  # SNR (dB)
    power_ary = 10 ** (power_db_ary / 10)

    timer = time.time()
    show_ary = list()
    for j, power in enumerate(power_ary):
        env.power = power

        episode_return = 0.0
        state = env.reset()
        for _ in range(env.max_step):
            # action = rd.randn(env.action_dim)
            action = env.get_action_mmse(state, bs_n, user_n, power, csi_noise_var)

            state, reward, done, info_dict = env.step(action)

            episode_return += reward
        episode_return /= env.max_step

        show_ary.append((power_db_ary[j], episode_return))

    show_ary = np.array(show_ary).round(3)
    print(show_ary)
    print(f"UsedTime: {time.time() - timer:8.3f}")
    '''
    [[-10.     -5.      0.      5.     10.     15.     20.     25.     30.   ]
     [  0.383   0.978   2.218   3.935   5.411   6.179   6.486   6.573   6.616]]
    UsedTime:   13.632
    env.max_step = 2 ** 12
    '''


def check__mmse_on_random_data():
    import time

    bs_n = 4  # antennas number of BaseStation
    user_n = 8  # user number
    sim_times = 2 ** 10  # simulate times
    csi_noise_var = 0.1  # channel state information

    power_db_ary = np.arange(-10, 30 + 5, 5)  # SNR (dB)
    power_ary = 10 ** (power_db_ary / 10)

    timer = time.time()
    show_ary = list()
    for j, power in enumerate(power_ary):
        '''generate state'''
        hall_ary = rd.randn(sim_times, user_n, bs_n) + rd.randn(sim_times, user_n, bs_n) * 1j
        hall_ary *= (1 / 2) ** 0.5
        hall_noise_ary = rd.randn(sim_times, user_n, bs_n) + rd.randn(sim_times, user_n, bs_n) * 1j
        hall_noise_ary *= (csi_noise_var / 2) ** 0.5

        sum_rate_temp = 0
        for t in range(sim_times):
            hall = hall_ary[t]
            hall_noisy = hall + hall_noise_ary[t]

            w_mmse = func_mmse(hall_noisy, bs_n, user_n, power, csi_noise_var)
            # '''check the power of w'''
            # power_temp = (w_mmse.real ** 2).sum() + (w_mmse.imag ** 2).sum()
            # assert power_temp < power + 0.0001

            sum_rate_temp += get_sum_rate(hall, w_mmse)

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


if __name__ == '__main__':
    check__mmse_on_env()
    # run_mmse_on_random_data()
    pass
