import numpy as np
import numpy.random as rd
# import matplotlib.pyplot as plt

from AgentRun import *

"""
BeamFormerEnvNoisy noise_std = 0.2
ceta0 power__l[1]
ceta1 power__l[3]
ceta2 power__l[5]

BeamFormerEnvNoisy noise_std = 0.0
ceta3 power__l[1]
ceta4 power__l[3]
beta1 power__l[5]
"""


class BeamFormerEnvNoisy:  # 2020-12-29
    def __init__(self, antennas_num=8, user_num=4,
                 max_power=1., noise_std=1.):
        """Down-link Multi-user MIMO beam-forming"""
        self.n = antennas_num
        self.k = user_num
        self.max_power = max_power  # P, max power of Base Station
        self.h_std = 1  # I, channel std
        self.y_var = 1  # sigma^2, symbol noise variant
        self.noise_std = noise_std  # gamma, observed channel noise std

        self.func_inv = np.linalg.inv
        self.h_k = None

        self.gamma_r = None
        self.now_step = None
        self.episode_return = None

        '''env information'''
        self.env_name = 'BeamFormerEnv-v0'
        self.state_dim = self.k * self.n
        self.action_dim = self.k * self.n
        self.if_discrete = False
        self.max_step = 2 ** 10

        self.r_offset = self.get__r_offset()  # behind '''env information'''
        # print(f"| BeamFormerEnv()    self.r_offset: {self.r_offset:.3f}")
        self.target_reward = self.r_offset * 1.5

    def reset(self):
        self.gamma_r = 0.0
        self.now_step = 0
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21

        self.h_k = rd.randn(self.k, self.n) * self.h_std
        state = self.h_k.reshape(self.state_dim) + rd.randn(self.state_dim) * self.noise_std
        return state.astype(np.float32)

    def step(self, action):
        w_k = action.reshape((self.k, self.n))
        rate_avg = self.get_sinr_rate(w_k, self.h_k)
        self.episode_return += rate_avg  # Compatibility for ElegantRL 2020-12-21
        r = rate_avg - self.r_offset

        self.h_k = rd.randn(self.k, self.n) * self.h_std
        state = self.h_k.reshape(self.state_dim) + rd.randn(self.state_dim) * self.noise_std

        self.now_step += 1
        done = self.now_step == self.max_step

        self.gamma_r = self.gamma_r * 0.99 + r
        if done:
            r = self.gamma_r
            self.episode_return /= self.max_step

        return state.astype(np.float32), r, done, {}

    def get_random_action(self):
        w_k = rd.randn(self.k, self.n)
        return self.action__power_limit(w_k, if_max=True)

    def get_traditional_action(self, state):  # h_k=state
        h_k = state.reshape((self.k, self.n))
        a_k = self.h_std ** 2 + np.dot(h_k.T, h_k) * (self.max_power / (self.k * self.y_var))
        # print(f'| traditional solution: a_k.shape={a_k.shape}')

        a_inv = self.func_inv(a_k + np.eye(self.n) * 1e-8)  # avoid ERROR: numpy.linalg.LinAlgError: Singular matrix
        # print(a_inv.shape, np.allclose(np.dot(a_k, a_inv), np.eye(self.n)))
        # print(f'| traditional solution: a_inv.shape={a_inv.shape}')

        w_k = np.dot(h_k, a_inv)
        # print(f'| traditional solution: a_inv__h_k.shape={a_inv__h_k.shape}')
        # action_k = a_inv__h_k * (self.max_p / (self.k * np.abs(a_inv__h_k).sum()))
        # print(f'| traditional solution: action_k.shape={action_k.shape}')

        return self.action__power_limit(w_k, if_max=True)

    def action__power_limit(self, w_k, if_max=False):  # w_k is action
        # print(f'| Power of BS: {np.power(w_k, 2).sum():.2f}')
        power = np.power(w_k, 2).sum() ** 0.5
        if if_max or power > self.max_power:
            w_k = w_k * (self.max_power ** 0.5 / power)

        # power = np.power(w_k, 2).sum() ** 0.5
        # print(f'| Power of BS: {power:.2f}')
        # print(f'| Power of BS: if Power < MaxPower: {power <= self.max_p}')
        return w_k

    def get_sinr_rate(self, w_k, h_k):
        hw_k = (w_k * h_k).sum(axis=1)
        h_w_k = np.dot(w_k, h_k.T).sum(axis=0)
        sinr_k = np.power(hw_k, 2) / (np.power(h_w_k - hw_k, 2) + self.y_var)
        # print(f'| Signal-to-Interference-and-Noise Ratio (SINR): shape={sinr_k.shape}')

        r_k = np.log(sinr_k + 1)
        # print(f'| rate of each user: shape={r_k.shape}')
        return r_k.mean()

    def get__r_offset(self):
        env = self
        self.r_offset = 0.0

        # traditional action
        state = env.reset()
        for i in range(env.max_step):
            action = env.get_traditional_action(state)
            state, reward, done, _ = env.step(action)
        r_avg__td = env.episode_return

        r_offset = r_avg__td
        return r_offset

    def get_snr_db(self):
        snr = self.max_power / self.y_var  # P/sigma^2
        return 10 * np.log10(snr)

    def show_information(self):
        print(f"| antennas_num, user_num: {self.n}, {self.k} "
              f"| SNR, max_power: {self.get_snr_db():.3f}, {self.max_power:.3f})\n")

    # def demo1229(self, action=None):
    #     w_k = rd.randn(4, 8)
    #     power1 = np.power(w_k, 2).sum() ** 0.5
    #     w_k = self.action__power_limit(w_k, if_max=True)
    #     power2 = np.power(w_k, 2).sum() ** 0.5
    #
    #     print('P:', power1, power2, self.max_power)
    #     exit()
    #     # w_k = self.get_random_action() if action is None else action  # w_k is action
    #
    #     '''Basic Station (BS)'''
    #     h_k = rd.randn(self.k, self.n) * self.h_std  # h_k is state
    #     print(f'| channel between BS and each user:  shape={h_k.shape}, h_std={self.h_std:.2f}')
    #     s_k = rd.randn(self.k)
    #     print(f'| symbol for each user from BS:      shape={s_k.shape}')
    #     x_n = np.dot(w_k.T, s_k)
    #     print(f'| transmitted signal   from BS:      shape={x_n.shape}')
    #
    #     # ## Tutorial of np.dot and np.matmul
    #     # a = np.ones((2, 3))
    #     # b = rd.rand(3, 4)
    #     # c = np.matmul(a, b) # (np.matmul == np.dot) when both a, b are 1D or 2D matrix
    #     # print(c)
    #     # y_k = [(h_k[i] * x_n).sum() for i in range(self.k)]
    #     # y_k = [np.dot(h_k[i], x_n.T) for i in range(self.k)]
    #     # y_k = [np.dot(x_n, h_k[i].T) for i in range(self.k)]
    #     # y_k = np.dot(h_k, np.expand_dims(x_n, 1)).T
    #     # y_k = np.dot(x_n, h_k.T)
    #     y_k = np.dot(x_n, h_k.T)
    #     print(f'| received signal by each user:      shape={y_k.shape}')
    #     noisy_y_k = y_k + rd.randn(self.k) * self.noise_y_std
    #     print(f'| received signal by each user:      shape={noisy_y_k.shape}, noise_y_std={self.noise_y_std:.2f}')
    #
    #     avg_r = self.get_sinr_rate(w_k, h_k)
    #     print(f'| rate of each user (random action): avg_r={avg_r:.2f}')
    #
    #     '''traditional solution'''
    #     action_k = self.get_traditional_action(h_k)
    #     avg_r = self.get_sinr_rate(action_k, h_k)
    #     print(f'| rate of each user (traditional):   avg_r={avg_r:.2f}')
    #
    #     '''traditional solution: noisy channel'''
    #     noisy_h_k = h_k + rd.randn(self.k, self.n) * self.noise_h_std
    #     action_k = self.get_traditional_action(noisy_h_k)
    #     avg_r = self.get_sinr_rate(action_k, h_k)
    #     print(f'| rate of each user (traditional):   avg_r={avg_r:.2f}, noise_h_std={self.noise_h_std:.2f}')


def simulate_random(env, sim_times=2 ** 18):
    # print(f'Random')
    # env = BeamFormerEnvNoisy(antennas_num, user_num, max_power, noise_h_std, noise_y_std)
    # env.show_information()

    import tqdm
    t = tqdm.trange(sim_times, desc='avgR')

    reward_sum1 = 0.0
    max_step = env.max_step

    for i in t:
        env.reset()
        for _ in range(max_step):
            action = env.get_random_action()  # env.get_traditional_action(state)
            env.step(action)

        reward_sum1 += env.episode_return

        t.set_description(f"avgR: {reward_sum1 / (i + 1):.4f}", refresh=True)
    reward_avg1 = reward_sum1 / sim_times
    # print(f"Random avgR: {reward_avg1:.4f}, SNR: {env.get_snr_db()} dB")
    return reward_avg1


def simulate_traditional(env, sim_times=2 ** 18):
    # print(f'Traditional')
    # env = BeamFormerEnvNoisy(antennas_num, user_num, max_power, noise_h_std, noise_y_std)
    # env.show_information()

    import tqdm
    t = tqdm.trange(sim_times, desc='avgR')

    reward_sum1 = 0.0
    max_step = env.max_step

    for i in t:
        state = env.reset()
        for _ in range(max_step):
            action = env.get_traditional_action(state)
            state, reward, done, _ = env.step(action)

        reward_sum1 += env.episode_return

        t.set_description(f"avgR: {reward_sum1 / (i + 1):.4f}", refresh=True)
    reward_avg1 = reward_sum1 / sim_times
    # print(f"Traditional avgR: {reward_avg1:.4f}, SNR: {env.get_snr_db()} dB")
    return reward_avg1


def run__plan():
    antennas_num = 8
    user_num = 4
    y_var = 1.0
    # max_power = 1.0
    noise_std = 0.2  # 1.0

    snr_db_l = np.linspace(-10, 20, 7)  # 13
    power__l = 10 ** (snr_db_l / 10) * y_var
    # print(snr_db_l.round(3))
    # print(power__l.round(3))

    sim_times = 2 ** 5  # * 2 ** 10

    recorder = list()

    for max_power in power__l:
        env = BeamFormerEnvNoisy(antennas_num, user_num, max_power, noise_std)

        r = simulate_traditional(env, sim_times)
        # r = simulate_random(env, sim_times)
        recorder.append(r)

    recorder = np.array(recorder).round(3)
    print('| Saved:', repr(recorder))


def train__rl():
    antennas_num = 8
    user_num = 4
    y_var = 1.0
    # max_power = 1.0
    noise_std = 0.0  # todo

    snr_db_l = np.linspace(-10, 20, 7)  # 13
    power__l = 10 ** (snr_db_l / 10) * y_var
    max_power = power__l[5]  # todo

    env = BeamFormerEnvNoisy(antennas_num, user_num, max_power, noise_std)

    from AgentZoo import AgentModSAC

    args = Arguments(rl_agent=AgentModSAC, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 2
    args.rollout_workers_num = 4
    args.if_break_early = False

    args.break_step = int(1e5 * 8)  # UsedTime: 900s (reach target_reward 200)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    print("| max_power:", max_power)
    print("| noise_std:", noise_std)


def data1229():
    antennas_num = 8
    user_num = 4
    noise_y_std = 1.0

    noise_h_std = 1.0
    max_power = 1.0

    snr_db_l = np.linspace(-10, 20, 7)  # 13
    print(snr_db_l.round(3))
    power__l = 10 ** (snr_db_l / 10) * (noise_y_std ** 2)
    print(power__l.round(3))

    xs = snr_db_l
    ys = np.array((1.97, 2.11,))  # ceta0
    ys = np.array((2.03, 2.11, 2.05, 2.32, 2.35, 2.25,))  # ceta0

    import matplotlib.pyplot as plt
    plt.title('MaxPower, antennasN=8, userK=4')

    ys = np.array([0.022, 0.06, 0.137, 0.26, 0.409, 0.557, 0.68])
    plt.plot(xs, ys, label='Random')
    ys = np.array([0.021, 0.058, 0.139, 0.279, 0.468, 0.676, 0.857])
    plt.plot(xs, ys, label='Traditional noise=1.0')
    ys = np.array([0.021, 0.059, 0.152, 0.329, 0.588, 0.89, 1.169])
    plt.plot(xs, ys, label='Traditional noise=0.5')
    ys = np.array([0.02, 0.06, 0.164, 0.382, 0.751, 1.224, 1.72])
    plt.plot(xs, ys, label='Traditional noise=0.2')
    ys = np.array([0.02, 0.06, 0.165, 0.4, 0.814, 1.403, 2.058])
    plt.plot(xs, ys, label='Traditional noise=0.1')
    ys = np.array([0.02, 0.06, 0.166, 0.406, 0.838, 1.509, 2.365])
    plt.plot(xs, ys, label='Traditional noise=0.0')

    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # run__plan()
    train__rl()
