import os
import time
import numpy as np
import torch as th
import pandas as pd
from typing import Union

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('Agg') if os.name != 'nt' else None  # Generating matplotlib graphs without a running X server [duplicate]
except ImportError:
    print(f"| ImportError import matplotlib",flush=True)
    mpl = None
    plt = None

TEN = th.Tensor
ARY = np.ndarray


class EncoderBase64:
    def __init__(self, encode_len: int):
        num_power = 6
        self.encode_len = encode_len
        self.string_len = -int(-(encode_len / num_power) // 1)  # ceil(num_nodes / num_power)

        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)
        assert self.base_num == 2 ** num_power

    def bool_to_str(self, x_bool: Union[TEN, ARY]) -> str:
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
        return x_str.zfill(self.string_len)

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
        x_bool = th.zeros(self.encode_len, dtype=th.bool)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.bool)
        return x_bool


class Evaluator:
    def __init__(self, save_dir: str, num_bits: int, x: TEN, v: float, if_maximize: bool):
        self.start_timer = time.time()
        self.recorder1 = []
        self.recorder2 = []
        encoder_base64 = EncoderBase64(encode_len=num_bits)  # todo plan to num_bits=x.shape[-1]
        self.bool_to_str = encoder_base64.bool_to_str
        self.str_to_bool = encoder_base64.str_to_bool

        self.best_x = x  # solution x
        self.best_v = v  # objective value of solution x

        self.if_maximize = if_maximize
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.record1(i=0, v=self.best_v)
        self.record2(i=0, vs=self.best_v, xs=self.best_x)

    def record1(self, i: float, v: float):
        self.recorder1.append((i, v))

    def record2(self, i: float, vs: Union[TEN, float], xs: TEN):
        if len(xs.shape) == 2:
            good_i = vs.argmax() if self.if_maximize else vs.argmin()
            good_x = xs[good_i]
            good_v = vs[good_i]
        else:
            good_x = xs
            good_v = vs
        good_v = float(good_v)

        exec_time = time.time() - self.start_timer
        self.recorder2.append((i, good_v, exec_time))

        if_update = (good_v > self.best_v) if self.if_maximize else (good_v < self.best_v)
        if if_update:
            self.best_x = good_x
            self.best_v = good_v
        return if_update

    def save_record_draw_plot(self, fig_dpi: int = 300):
        if len(self.recorder1) == 0 or len(self.recorder2) == 0:
            return

        '''save_record'''
        recorder1 = np.array(self.recorder1)
        recorder2 = np.array(self.recorder2)
        np.save(f"{self.save_dir}/recorder1.npy", recorder1)
        np.save(f"{self.save_dir}/recorder2.npy", recorder2)

        '''draw_plot'''
        if plt is None:
            return
        plt.plot(recorder1[:, 0], recorder1[:, 1], linestyle='-', label='real time')
        plt.plot(recorder2[:, 0], recorder2[:, 1], linestyle=':', label='back test')
        plt.scatter(recorder2[:, 0], recorder2[:, 1], s=24)

        plt.title(f"best_obj_value_{self.best_v}")
        plt.axis('auto')
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.save_dir}/recorder.jpg", dpi=fig_dpi)
        plt.close('all')

    def logging_print(self, show_str: str = '', if_show_x: bool = False):
        used_time = int(time.time() - self.start_timer)
        x_str = self.best_x_str if if_show_x else ''
        x_str = f"{x_str}" if len(x_str) > 32 else x_str
        i = self.recorder2[-1][0]

        log_str = f"|{i:6} {used_time:4} sec  best {self.best_v:12.4f} {show_str}  x_str: {x_str}"
        print(log_str, flush=True)
        return log_str

    @property
    def first_v(self) -> float:
        return self.recorder2[0][1]

    @property
    def best_x_str(self):
        return self.bool_to_str(self.best_x).replace('\n', '')


class Recorder:
    def __init__(self):
        self.recorder = []
        self.start_time = time.time()

    def add_and_print(self, repeat_id, buffer_id, objective_item):
        exec_time = int(time.time() - self.start_time)
        self.recorder.append((repeat_id, buffer_id, exec_time, objective_item))
        print(f"| {repeat_id:4} {buffer_id:4}  {exec_time:4} sec | obj {objective_item:9.4f}", flush=True)

    def save_as_csv(self, csv_path: str = './recorder.csv'):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.remove(csv_path) if os.path.exists(csv_path) else None
        df = pd.DataFrame(self.recorder, columns=['repeat_id', 'buffer_id', 'exec_time', 'objective'])
        df.to_csv(csv_path, index=False)

    @staticmethod
    def plot_training_recorder(csv_path, ignore_n: int = 8):
        # data_dir = f"./recorder"
        dir_path = os.path.dirname(csv_path)
        file_name = os.path.basename(csv_path)

        df = pd.read_csv(csv_path)
        df = df.iloc[min(ignore_n, df.shape[0] // 20):]

        name = os.path.splitext(file_name)[0]
        plt.plot(df['exec_time'], df['objective'], marker=None)
        plt.xlabel('exec_time')
        plt.ylabel('objective')
        plt.title(name)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dir_path}/{name}.jpg", dpi=200)
        print(f"| TrainingLogger save jpg in {dir_path}/{name}.jpg", flush=True)

    @staticmethod
    def plot_training_recorders(data_dir: str = './recorder', graph_type_id: int = 0):
        # data_dir = f"./recorder"
        import re

        fig, axs = plt.subplots(2, 4, figsize=(18, 10))
        axs = axs.flatten()

        ignore_n = 8
        graph_type = ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw'][graph_type_id]
        file_names = os.listdir(data_dir)
        file_names = [name for name in file_names if name.find(graph_type) >= 0]
        file_names = sorted(file_names, key=lambda name: int(re.search(r'Node(\d+)', name).group(1)))
        for i, name in enumerate(file_names):
            path = f"{data_dir}/{name}"
            df = pd.read_csv(path)

            df = df.iloc[ignore_n:]
            ax = axs[i]
            ax.plot(df['exec_time'] / 3600, df['objective'], marker=None)
            ax.set_xlabel('exec_time (hours)')
            ax.set_ylabel('objective')
            ax.set_title(f'{name}')
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def read_info_from_recorder(recorder, per_second):
    recorder = np.array(recorder)
    exec_times = recorder[:, 2]
    obj_values = recorder[:, 1]

    for i in range(1, obj_values.shape[0]):
        obj_values[i] = obj_values[i - 1:i + 1].max()

    ids = []
    for exec_time in range(per_second, int(exec_times[-1]), per_second):
        idx = np.searchsorted(exec_times, exec_time, side='right') - 1
        ids.append(idx)
    ids.append(-1)
    ids = np.array(ids)

    info_str = ' '.join(str(v) for v in obj_values[ids].astype(int))
    return info_str


'''check'''


def check_evaluator():
    gpu_id = 0
    graph_name, num_nodes = 'gset_14', 800

    temp_xs = th.zeros((1, num_nodes))
    temp_vs = th.ones((1,))

    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_bits=num_nodes, if_maximize=True,
                          x=temp_xs[0], v=temp_vs[0].item(), )
    assert isinstance(evaluator, Evaluator)


def check_recorder():
    net_path = "./model/graph_trs_gset_14_800_ID0.pth"
    csv_path = f"{net_path}.csv"

    recorder = Recorder()
    recorder.plot_training_recorder(csv_path=csv_path, ignore_n=16)


X_G14 = """ 11Re2ycMx2zCiEhQl5ey$HyYnkUhDVE6KkPnuuhcWXwUO9Rn1fxrt_cn_g6iZFQex1YpwjD_j7KzbNN71qVekltv3QscNQJjrnrqHfsnOKWJzg9nJhZ$qh69 $X_BvBQirx$i3F """
# 3064, SOTA=3064
X_G14_1 = """ 11Re2ydMx2zCiEhQl5ey$PyYnkUhDVE6KkQnuuhc0XwUO9RnXfxrt_dn_g6aZFQ8x1YpwbD_j7KzaNN71qVuklpv3Q_cNQJjnnrrHjsnOKWIzg9nJxZ$qh69 $n_BHBRirx$i3F """
# 3064, SOTA=3064
X_G14_2 = """ 2_aNz3Of4z2pJnKaGwN30k3TEHXKoWnvhHaE77KPlU5XdsaE_UCA81PE1LvJSmbN4_Ti5Qo1IOh2Aeeu_BWNHGC6yb1GebiIAEAAkI9EdhVj2LsEiKS2BKvs 0E1qkqaJ840Jym """
# 3064, SOTA=3064

X_G15 = """ hzvKByHMl4xek23GZucTFBM0f530k4DymcJ5QIcqJyrAoJBkI3g5OaCIpvGsf$l4cLezTm6YOtuDvHtp38hIwUQc3tdTBWocjZj5dX$u1DEA_XX6vESoZz2W NZpaM3tN$bzhE """
# 3050, SOTA=3050
""" 3K26hq3kfGx4N1zylS7HYmqf$Mwy$Hxo3FPiubjPBiBgrDirHj_LwZRpjC6l8I0GxPgN2YBvTb87oMkeCytKj5pbPy8OYyPDPIS2wOS27_onr1UUP6pZDCAV VeSCVfyCe2Q0Kn """
# 3050, SOTA=3050
""" 3K26hq3kfGx4N1zylS7PYmqf$Mwy$nxo3FPiwbjPBi3ArDirHjyLwdRpjC6l8M0mxPgN2YFvTb87o4k8CStKj5fbPyCOYqRDPISIwOUA7_onr1UUn6vZDCAV VeSCRfy8eAQ3Sn """
# 3050, SOTA=3050

X_G49 = """ LLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggg gggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLL LLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLgggggggggg ggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLL LLLQgggggggggggggggg """
# 6000, SOTA=6000
""" ggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLL LLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQgggggg ggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLL LLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggg gggbLLLLLLLLLLLLLLLL """
# 6000, SOTA=6000

X_G50 = """ LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL gggggggggggggggggggg """
# 5880, SOTA=5880
""" LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL gggggggggggggggggggg """
# 5880, SOTA=5880

X_G22 = """ 3NbGRdQwo$UleJIHz3aimCPRiK5a9$y_7l3rCmgS6prQwKeXyJ2V9uJePA7WwL4_Eqx37mJaVUNE9V6qrXw1cr4Q0Ozv22bvkqee9QEAGcV5DsT0TNCcJ$RG 9wGK3$TxE4j6PYXgxdqIaXPGScKsPj3BvpxdNn3Wfy3tfL9H3zddbHofnQ0bMLX5AQEBRb5gki2YZ1kuwTlgc9l1p_qZfuSUvPf2DWx4nhMFYgQ3NleSc77S XSSzTD9m6VMKrfbn8CbZGWtwsUkQXb3UW6JnnARtR5XaZrW$x4NQ52LiVrEZpFIQnzPsfv8utMCNptTsanvIvZQ0026wJG """
# 13359, SOTA=13359

X_G55 = ''' 2VSaO16Jc5tj7YbROV8ZsS4R6m5PSMJDQRVzXdCBqJc7Du$XK55JVOglmHtF2Xw7qqqGXtsyw9OpSXWskTosv8Hwro9u3u9JrSFYuPNDu_wqeD4$sInDTIBZ wWNqalEad7ykkQ2SrHPscKoPcWc9mjmWjg5YOuqP937QKh2C9uDJ1byHt9OG6qAJcd9WYCIxc8Ee$7u8vNJhMJbEzo_t0tEZRFK92lxFTwo2iayB1W12EqS3 qUYBf1dUuykJDPVN8xTIVCESaKm7FQqNpgByD7nkLxQkbP34o$IW6StOnQo5KYUrqL_E4mIAl4R1V2ItVj75gc9hUdJfkPofUAfbIRikFFKKb28n1rkes6qw 9GYzWsS961TLyuwyvaK6ADeGOQ0wBGzCDpzy20gP0dEPLEJaaGqiUDSIQx9FB07KkA55LBk4baeZDmGEZ4rFIqZtou26Mhf9vOcprBStL$XxtsYZZmwOHrxY oqbX5b_IFpcCE4$gmttK3U_xomeYt2vRQkl7eZHx4mrU9nkgtVvPIiQiLTic0OIep53enkZ56TNjppoiQo_JWhYQyq_ePyhA4F0zRuK0hwxEAFMRmPTaRl$g Oe1Y27pBxiv83r3qmevxWyhhNoHJzutn2FRWX$1aQykiSXk9mF8nWEjLj32UXf84MEyzxnw6LFJdXmmYnWszaHoA39znnnRosB1jBL9KzidrHTZB0tac2ArQ ONgrOwS7xpnNnozGfvQx_2327RWIlGgWQKJdvuti3_kaFUzAGg$ZEERsCYe7X$AA6xxqYhnAkCAkF68HIN7GrMbqlxhMme4l9260sS82G5TbRzL$iz '''
# 10298, SOTA=10296

X_G70 = """ 5gXwIRKvLB05LYYX9scVmMZzDq362k5LduXt9$mSCZoc6eNg_rKEgbFOhJi_dDe38G1u2cpZsM4vhSzdijcVsXFT_mLiN5pOIqAfOzTPoaP_8GkJR446APTf JSaXkSvIxVhcUB7c26cze9ze4uag7JgycspwV_prZrp_IwKkfPrbP8KFz5YsJgg97EgY2WZ_4j7gpD1Ax$ycOXzazq_e8fqKmJAHlubFNZRBlBU5vsNl1AiS TnsWp8CVl4xNfaqD84qgIFSQf3XpZM2sE8Bi9oW23vQ9B0TASbBjJca18aoPXSi3N6m82lvZz6WaIRD0nylh7aqOtOj1lrZBKvlj1BSJDW383LYEkX26Zt_r VuDuI$ZchT_xn9Yedh2oBN8K2CXujPqDWTGfh5PnyuQg6uT2fnXMRY0NQOLNYxOHQ0wljzbgVxlA$TL9lN9fofBV09Vi2zuB$yJMq4NQE8C5HjfsXrU9iHg6 DFVJk_9qujCWXlN3hD8OZXVKYaF_iiIwRpW4UxFGzcE1sRd3bb1Df_xIBZQuwGvUWVOrRVRufcMaJ$_NLdX2MFjxNA4BHBXLtFgYRALLAN$XFsTdUE_HtGBu I356XIISHudo_giKlLAkcw1i_5KcdhVXrGQt0sWS3$V19gw1TQYOdL3jt0EfN7gITEWcu7Vaw7BEIZ$zNd23v7lsWUPAXrmif$dfLgpEwr_OSPSyAqiwXoAL qMIwFJR5mipfTRTimCKXj0SJd97r8Qo3oAHkACEhlhIKqmuyrF7FTV253CkDvIRBRuNf1SX7mzV3u2FvDoxhEcYb2OudX$fQS8a4q6rQh_skBG24JRabm6_d wI5ZLEhqGbo6gkovvAG5vQUq9RP4Ioob478_JTzneUavNeyyAdE7cuKsdpXfeEav3_NPvVUuvvxMa6xaqs4_Sv41jUiosnFGU7qW_AGPYy05gnviJ8rKG_5v vroYNXHg0fglzYHwL5DslLMBhz0xNnxp295hzAHp7On7607VHh_YQxIv$Dc4lz4Xn2qXr7L6gqAJxAkQZwmb4dFMum$JQ1fgjS_DnOabbRfn9ai3QFKky$Gc rHiaBMYneaBTjd92EQv_4aOOxSIzIEJVtNLPv2clXf8SB984lz4rMrgCEIx_sXOtqGSFScSemRmMiTJZQiZg3UXUxNhJiGvVa8eLwjyOC_8N6IsnUrxk8GRw 8Ja31nPwWsShEMpjr_6AZF2gXezxBgubdXh98CjtMTQCO4FL50oSwBP7tX6FmsLDBej$ke6fRCQgQW4XZox0kbdQ3fFy6w_87leq2qQ75hCQatL7zZPluwGH QeEKijfDNsnZgFl5KSf$j_Kh25no$HUp0$K50Jkpha4eXfNV0yasGov3bQ5L5GPYGAaMLluqOMmthGgSenLan_cd90BRWGfXQJJj64h88KmIwjAXKrNmFxWw YPWO1chl8zx20J7$gCmvASk4jH9pJKkX8RyHEH74cwS9pWjmywHbAmg7t54QwfcFvfLyzTrziJaz2oHyj03ypTZWc1__SuW8wfUXfvV1tU86DxLVnurcJfO1 hI1MHL$$n7W32E96659blS3WAnnGOr0Vwg7MMvyKS8ignmH_pfy7g1TeTVF1R7SSnUPCojEBO7Sz4ds6OcGu2QfLzCMcMg4SRJho4RueZxm """
# 9583, SOTA=9595







if __name__ == '__main__':
    # check_evaluator()
    check_recorder()
    # check_solution_x()
    # check_solution_x_via_load_graph_info_from_data_dir()
