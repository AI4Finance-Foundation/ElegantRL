import os
import time
import math
import numpy as np
import torch as th

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('Agg') if os.name != 'nt' else None  # Generating matplotlib graphs without a running X server [duplicate]
except ImportError:
    mpl = None
    plt = None

TEN = th.Tensor


class EncoderBase64:
    def __init__(self, num_nodes: int):
        num_power = 6
        self.num_nodes = num_nodes
        self.num_length = -int(-(num_nodes / num_power) // 1)  # ceil(num_nodes / num_power)

        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)
        assert self.base_num == 2 ** num_power

    def bool_to_str(self, x_bool: TEN) -> str:
        x_int = int(''.join([('1' if i else '0') for i in x_bool.tolist()]), 2)

        '''bin_int_to_str'''
        base_num = len(self.base_digits)
        x_str = ""
        while True:
            remainder = x_int % base_num
            x_str = self.base_digits[remainder] + x_str
            x_int //= base_num
            if x_int == 0:
                break

        if len(x_str) > 120:
            x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
        if len(x_str) > 64:
            x_str = f"\n{x_str}"
        return x_str.zfill(self.num_length)

    def str_to_bool(self, x_str: str) -> TEN:
        x_b64 = x_str.replace('\n', '').replace(' ', '')

        '''b64_str_to_int'''
        x_int = 0
        base_len = len(x_b64)
        for i in range(base_len):
            digit = self.base_digits.index(x_b64[i])
            power = base_len - 1 - i
            x_int += digit * (self.base_num ** power)

        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.bool)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.bool)
        return x_bool


class Evaluator:
    def __init__(self, save_dir: str, num_nodes: int, x: TEN, v: float):
        self.start_timer = time.time()
        self.recorder1 = []
        self.recorder2 = []
        self.encoder_base64 = EncoderBase64(num_nodes=num_nodes)

        self.best_x = x  # solution x
        self.best_v = v  # objective value of solution x

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.record1(i=0, v=self.best_v)
        self.record2(i=0, v=self.best_v, x=self.best_x)

    def record1(self, i: float, v: float):
        self.recorder1.append((i, v))

    def record2(self, i: float, v: float, x: TEN):
        self.recorder2.append((i, v))

        if_update = v > self.best_v
        if if_update:
            self.best_x = x
            self.best_v = v
        return if_update

    def plot_record(self, fig_dpi: int = 300):
        if plt is None:
            return

        if len(self.recorder1) == 0 or len(self.recorder2) == 0:
            return
        recorder1 = np.array(self.recorder1)
        recorder2 = np.array(self.recorder2)
        np.save(f"{self.save_dir}/recorder1.npy", recorder1)
        np.save(f"{self.save_dir}/recorder2.npy", recorder2)

        plt.plot(recorder1[:, 0], recorder1[:, 1], linestyle='-', label='real time')
        plt.plot(recorder2[:, 0], recorder2[:, 1], linestyle=':', label='back test')
        plt.scatter(recorder2[:, 0], recorder2[:, 1])

        plt.title(f"best_obj_value {self.best_v}")
        plt.axis('auto')
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.save_dir}/recorder.jpg", dpi=fig_dpi)
        plt.close('all')

    def logging_print(self, x: TEN, v: float, show_str: str = '', if_show_x: bool = False):
        used_time = int(time.time() - self.start_timer)
        x_str = self.encoder_base64.bool_to_str(x) if if_show_x else ''
        i = self.recorder2[-1][0]
        print(f"| UsedTime {used_time:8}  i {i:8}  best_v {self.best_v:9.2e}  good_v {v:9.2e}  "
              f"{show_str}  {x_str}")


def check_evaluator():
    gpu_id = 0
    graph_name, num_nodes = 'gset_14', 800

    temp_xs = th.zeros((1, num_nodes))
    temp_vs = th.ones((1,))

    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())
