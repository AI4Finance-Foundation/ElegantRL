import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def beta_2020_1010():
    cwd = ['/home/yonv/code/ElegantRL',
           '/home/yonv/code/ElegantRL_cwd/2020-09-09'][1]

    ary_list = list()
    max_rs = list()
    min_rs = list()
    import os
    for path in Path(f"{cwd}").rglob('LunarLander-v2*'):
        print(path)

        path = f"{path}/record_evaluate.npy"
        if not os.path.exists(path):
            continue
        ary = np.load(path)
        print('ary.shape ==', ary.shape)

        # xs = ary[:, 0]
        # ys = ary[:, 1]
        # plt.plot(xs, ys)
        # plt.show()
        ary_list.append(ary[:, :2])

        max_rs.append(ary[:, 1].max())
        min_rs.append(ary[:, 1].min())

    print(';;', max(max_rs), min(min_rs))
    exit()
    ary = np.vstack(ary_list)
    print(ary.shape)

    sort_idx = ary[:, 0].argsort()
    ary = ary[sort_idx]

    num = 128
    short_ary = list()
    ary_len = int(ary[-1, 0])
    gap = ary_len // num
    prt0 = prt1 = 0
    for i in range(0, ary_len, gap):
        while ary[prt1, 0] < i:
            prt1 += 1
        if prt1 < prt0 + 1:
            continue

        part = ary[prt0:prt1]
        x_avg, y_avg = part.mean(axis=0)
        y_ary = part[:, 1]
        # y_std_up = y_ary[y_ary > y_avg].std()
        # y_std_dw = y_ary[y_ary < y_avg].std()
        y_std_up = (np.abs(y_ary[y_ary > y_avg] - y_avg)).mean()
        y_std_dw = (np.abs(y_ary[y_ary < y_avg] - y_avg)).mean()

        short_ary.append((x_avg, y_avg, y_std_up, y_std_dw))

        prt0 = prt1

    short_ary = np.array(short_ary)

    xs = short_ary[:, 0]
    ys = short_ary[:, 1]
    ys_up = short_ary[:, 2]
    ys_dw = short_ary[:, 3]
    plt.plot(xs, ys)
    plt.fill_between(xs, ys + ys_up, ys - ys_dw, alpha=0.3)

    plt.plot(ary[:, 0], ary[:, 1], alpha=0.2, color='coral')

    plt.show()


# beta_2020_1010()

