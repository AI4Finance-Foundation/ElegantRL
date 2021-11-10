import sys

from elegantrl.env import build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp
from elegantrl.agent import AgentStep1AC, AgentShareStep1AC


def demo_down_link_task():
    env_name = ['DownLinkEnv-v0', 'DownLinkEnv-v1'][ENV_ID]
    agent_class = [AgentStep1AC, AgentShareStep1AC][DRL_ID]
    args = Arguments(env=build_env(env_name), agent=agent_class())
    args.random_seed += GPU_ID

    args.net_dim = 2 ** 8
    args.batch_size = int(args.net_dim * 2 ** -1)

    args.max_memo = 2 ** 17
    args.target_step = int(args.max_memo * 2 ** -4)
    args.repeat_times = 0.75
    args.reward_scale = 2 ** 2
    args.agent.exploration_noise = 2 ** -5

    args.eval_gpu_id = GPU_ID
    args.eval_gap = 2 ** 9
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1

    args.learner_gpus = (GPU_ID,)

    if_use_single_process = 0
    if if_use_single_process:
        train_and_evaluate(args, )
    else:
        args.worker_num = 4
        train_and_evaluate_mp(args, )


"""
AgentStep1AC

| Remove cwd: ./AgentStep1AC_DownLinkEnv-v0_(1,)
################################################################################
ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  6.55e+04  708.26 |  708.26    5.9   1023     0 |    0.69   0.02  -0.68
0  1.34e+06  857.54 |  857.54    2.9   1023     0 |    0.84   0.01  -0.55
0  2.20e+06  890.26 |  890.26   10.5   1023     0 |    0.88   0.01  -0.65
0  2.62e+06 1071.47 | 1071.47    6.4   1023     0 |    1.02   0.02  -0.64
0  5.18e+06 1162.44 | 1162.44    5.1   1023     0 |    1.12   0.02  -0.68
0  7.73e+06 1303.46 | 1303.46    2.4   1023     0 |    1.32   0.02  -0.64
0  1.00e+07 1619.88 | 1619.88    6.2   1023     0 |    1.55   0.02  -0.73
0  1.51e+07 1687.30 | 1687.30   22.1   1023     0 |    1.65   0.02  -0.81
0  2.01e+07 1822.41 | 1714.13    0.0   1023     0 |    1.67   0.02  -0.90
0  2.53e+07 1822.41 | 1623.96    0.0   1023     0 |    1.56   0.02  -0.88
0  3.21e+07 2098.69 | 2098.69    4.4   1023     0 |    2.03   0.02  -0.97


AgentShareStep1AC

| Remove cwd: ./AgentShareStep1AC_DownLinkEnv-v1_(2,)
################################################################################
ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  6.55e+04  711.66 |
0  6.55e+04  711.66 |  711.66    3.3   1023     0 |    2.79   0.68  -1.83

| DownLinkEnv:    2.000    currR  777.302    r_mmse  982.375    r_ones  162.081
0  6.23e+05  884.06 |
0  6.23e+05  884.06 |  884.06   11.2   1023     0 |    3.15   0.97  -2.85

| DownLinkEnv:    3.000    currR  909.626    r_mmse 1132.550    r_ones  240.852
0  1.18e+06 1090.63 |
0  1.18e+06 1090.63 | 1090.63    1.0   1023     0 |    3.24   0.73  -2.85

| DownLinkEnv:    4.000    currR  884.971    r_mmse 1072.535    r_ones  322.279
0  1.74e+06 1099.45 |
0  1.74e+06 1099.45 | 1099.45   20.2   1023     0 |    3.24   0.73  -2.90

| DownLinkEnv:    5.000    currR 1199.955    r_mmse 1464.310    r_ones  406.891
0  2.29e+06 1150.17 |
0  2.29e+06 1150.17 | 1150.17   11.0   1023     0 |    3.21   0.79  -2.88
0  2.85e+06 1150.17 | 1005.63    0.0   1023     0 |    3.21   1.01  -2.92
0  3.41e+06 1150.17 | 1145.57    0.0   1023     0 |    3.28   0.79  -2.96
0  3.96e+06 1233.29 |
0  3.96e+06 1233.29 | 1233.29    3.4   1023     0 |    3.19   0.73  -3.00

| DownLinkEnv:    6.000    currR 1505.995    r_mmse 1846.890    r_ones  483.309
0  4.52e+06 1302.20 |
0  4.52e+06 1302.20 | 1302.20    5.4   1023     0 |    3.21   0.81  -3.04
0  5.08e+06 1453.68 |
0  5.08e+06 1453.68 | 1453.68    2.4   1023     0 |    3.25   0.59  -3.01
0  5.64e+06 1468.65 |
0  5.64e+06 1468.65 | 1468.65   20.9   1023     0 |    3.30   0.80  -3.08
0  6.19e+06 1579.85 |
0  6.19e+06 1579.85 | 1579.85   20.0   1023     0 |    3.74   0.74  -3.15

| DownLinkEnv:    7.000    currR 1836.257    r_mmse 2259.061    r_ones  567.847
0  6.75e+06 1783.22 |
0  6.75e+06 1783.22 | 1783.22    5.5   1023     0 |    4.41   0.73  -3.16
0  7.31e+06 1783.22 | 1768.64   20.3   1023     0 |    4.51   0.88  -3.23
0  7.86e+06 1783.22 | 1719.81    0.0   1023     0 |    4.62   0.64  -3.27
0  8.42e+06 1978.54 |
0  8.42e+06 1978.54 | 1978.54   10.6   1023     0 |    4.27   0.59  -3.96
0  8.98e+06 2003.92 |
0  8.98e+06 2003.92 | 2003.92   28.3   1023     0 |    4.83   0.55  -3.94
0  9.50e+06 2010.96 |
0  9.50e+06 2010.96 | 2010.96    8.2   1023     0 |    5.04   0.63  -4.02
0  1.01e+07 2010.96 | 1994.47    0.0   1023     0 |    5.20   0.49  -3.96
0  1.06e+07 2013.73 |
0  1.06e+07 2013.73 | 2013.73   18.9   1023     0 |    5.67   0.57  -4.06
"""

"""2021-11-10
GPU 83  GPU 2  MLP      RP_TIM=0.75 T_STEP=-4  B_SIZE=net_dim*1    1927, 10e6, 21e6 
GPU 83  GPU 3  DenseNet Env-v1 Step1AC
GPU 83  GPU 4  DenseNet B_SIZE=net_dim*0.5 NOISE=-5                1946, 10e6, 19e6 
GPU 83  GPU 0  DenseNet Env-v1 ShareStep1AC lr=1.1
GPU 83  GPU 1  DenseNet Env-v1 ShareStep1AC lr=1.25

GPU 111 GPU 1  DenseNet Env-v1 ShareStep1AC lr=0.75
GPU 111 GPU 2  DenseNet Env-v1 ShareStep1AC lr=1.50                2010, 06e6, 08e6
GPU 111 GPU 3  DenseNet Env-v1 ShareStep1AC lr=2.00
"""

GPU_ID = int(eval(sys.argv[1]))
DRL_ID = int(eval(sys.argv[2]))
ENV_ID = 1  # int(eval(sys.argv[2]))

SHA_LR = float(eval(sys.argv[3]))

if __name__ == '__main__':
    dir(sys)
    # check_network()
    demo_down_link_task()
