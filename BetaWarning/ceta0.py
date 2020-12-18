import numpy as np
import numpy.random as rd

"""
2020-12-18 12:00 Base and Extension 1
"""


class BeamFormer:
    def __init__(self, antennas_num=20, user_num=10):  # antennas_num=20 user_num=10
        """Down-link Multi-user MIMO beam-forming"""
        self.n = antennas_num
        self.k = user_num

        self.max_p = 2  # P, max power of Base Station
        self.h_std = 1  # I, std of channel
        self.noise_y_std = 1.0  # sigma, std of received channel noise
        self.noise_h_std = 1.0  # gamma, std of observed channel noise

        self.func_inv = np.linalg.inv

    def get_snr(self):
        return self.max_p / self.noise_y_std ** 2  # P/sigma^2

    def get_sinr_rate(self, w_k, h_k):
        hw_k = (w_k * h_k).sum(axis=1)
        h_w_k = np.dot(w_k, h_k.T).sum(axis=0)
        sinr_k = np.power(hw_k, 2) / (np.power(h_w_k - hw_k, 2) + np.power(self.noise_y_std, 2))
        # print(f'| Signal-to-Interference-and-Noise Ratio (SINR): shape={sinr_k.shape}')
        r_k = np.log(sinr_k + 1)
        # print(f'| rate of each user: shape={r_k.shape}')
        return r_k.mean()

    def get_random_action(self):
        w_k = rd.randn(self.k, self.n)
        return self.action__power_limit(w_k, if_max=True)

    def get_traditional_action(self, h_k):
        hh_k = np.dot(h_k.T, h_k)
        # print(f'| traditional solution: hh_k.shape={hh_k.shape}')
        a_k = self.h_std ** 2 + hh_k * (self.max_p / (self.k * self.noise_y_std ** 2))
        # print(f'| traditional solution: a_k.shape={a_k.shape}')

        a_inv = self.func_inv(a_k + np.eye(self.n) * 1e-8)  # avoid ERROR: numpy.linalg.LinAlgError: Singular matrix
        # print(a_inv.shape, np.allclose(np.dot(a_k, a_inv), np.eye(self.n)))
        # print(f'| traditional solution: a_inv.shape={a_inv.shape}')

        a_inv__h_k = np.dot(h_k, a_inv)
        # print(f'| traditional solution: a_inv__h_k.shape={a_inv__h_k.shape}')
        # action_k = a_inv__h_k * (self.max_p / (self.k * np.abs(a_inv__h_k).sum()))
        # print(f'| traditional solution: action_k.shape={action_k.shape}')
        return self.action__power_limit(a_inv__h_k, if_max=True)

    def action__power_limit(self, w_k, if_max=False):  # w_k is action
        # print(f'| Power of BS: {np.power(w_k, 2).sum():.2f}')
        power = np.power(w_k, 2).sum() ** 0.5
        if if_max or power > self.max_p:
            w_k = w_k / (power / self.max_p ** 0.5)

        # power = np.power(w_k, 2).sum() ** 0.5
        # print(f'| Power of BS: {power:.2f}')
        # print(f'| Power of BS: if Power < MaxPower: {power <= self.max_p}')
        return w_k

    def demo0(self, action=None):
        w_k = self.get_random_action() if action is None else action  # w_k is action

        '''Basic Station (BS)'''
        h_k = rd.randn(self.k, self.n) * self.h_std  # h_k is state
        print(f'| channel between BS and each user: channel shape={h_k.shape}, std={self.h_std:.2f}')
        s_k = rd.rand(self.k)
        print(f'| symbol for each user from BS: shape={s_k.shape}')
        x_n = np.dot(w_k.T, s_k)
        print(f'| transmitted signal from BS:   shape={x_n.shape}')

        # ## Tutorial of np.dot and np.matmul
        # a = np.ones((2, 3))
        # b = rd.rand(3, 4)
        # c = np.matmul(a, b) # (np.matmul == np.dot) when both a, b are 1D or 2D matrix
        # print(c)
        # y_k = [(h_k[i] * x_n).sum() for i in range(self.k)]
        # y_k = [np.dot(h_k[i], x_n.T) for i in range(self.k)]
        # y_k = [np.dot(x_n, h_k[i].T) for i in range(self.k)]
        # y_k = np.dot(h_k, np.expand_dims(x_n, 1)).T
        # y_k = np.dot(x_n, h_k.T)
        y_k = np.dot(x_n, h_k.T)
        print(f'| received signal by each user: signal: shape={y_k.shape}')
        noisy_y_k = y_k + rd.randn(self.k) * self.noise_y_std
        print(f'| received signal by each user: signal: shape={noisy_y_k.shape}, noise std={self.noise_y_std:.2f}')

        avg_r = self.get_sinr_rate(w_k, h_k)
        print(f'| rate of each user (random action): avg={avg_r:.2f}')

        '''traditional solution'''
        action_k = self.get_traditional_action(h_k)
        avg_r = self.get_sinr_rate(action_k, h_k)
        print(f'| rate of each user (traditional)  : avg={avg_r:.2f}')

        '''traditional solution: noisy channel'''
        noisy_h_k = h_k + rd.randn(self.k, self.n) * self.noise_h_std
        action_k = self.get_traditional_action(noisy_h_k)
        avg_r = self.get_sinr_rate(action_k, h_k)
        print(f'| rate of each user (traditional)  : avg={avg_r:.2f}')


if __name__ == '__main__':
    env = BeamFormer()
    env.demo0()
