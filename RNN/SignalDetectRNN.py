import os
import time

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

GPU_id = 0
Mod_dir = 'SignalDetect_{}'.format(GPU_id)
Mu = np.array([0.6, 0.8, 1.2, 0.9, 0.7, 0.5, 1.1, 0.3, 0.7, 1.1])  # arbitrary mean
Inp_dim = 10
Mid_dim = 10  # 24
Out_dim = 1


class RegLSTM(nn.Module):
    def __init__(self, input_size, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression
        # self.reg = nn.Linear(mod_dim, out_dim)  # regression

    def forward(self, x):
        x, hc = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)

        x = x.view(seq_len, batch_size, -1)
        return x


def calculate_avg_std(y_values, smooth_kernel=32):
    y_reward = np.array(y_values)
    r_avg = list()
    r_std = list()
    for i in range(len(y_reward)):
        i_beg = i - smooth_kernel // 2
        i_end = i_beg + smooth_kernel

        i_beg = 0 if i_beg < 0 else i_beg
        rewards = y_reward[i_beg:i_end]
        r_avg.append(np.average(rewards))
        r_std.append(np.std(rewards))
    r_avg = np.array(r_avg)
    r_std = np.array(r_std)
    return r_avg, r_std


def run_train():
    seq_len = 512
    batch_size = 64
    train_epoch = 2 ** 10  # 12

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(Inp_dim, Out_dim, Mid_dim, mid_layers=2).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''data'''
    with torch.no_grad():
        mask01 = torch.zeros((seq_len, batch_size, Out_dim), dtype=torch.float32, device=device)
        i_beg = (seq_len - batch_size) // 2
        for i in range(batch_size):
            mask01[i_beg + i:, i, 0] = 1
        mask10 = -mask01 + 1

        mu = torch.tensor(Mu, dtype=torch.float32, device=device)
        mu = mu.view(1, 1, Inp_dim)
        ten_mean = mu * torch.ones((seq_len, batch_size, Inp_dim), device=device)
        ten_mean = ten_mean * mask01

        # print(train_x.size())
        # print(train_y.size())
        # print(ten_mean.size())

    '''train'''
    with torch.no_grad():
        weights = np.tanh(np.arange(seq_len) * (np.e / seq_len))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    train_y = torch.ones((seq_len, batch_size, Out_dim), dtype=torch.float32, device=device)
    train_y = -train_y * mask10 + train_y * mask01

    """training loop"""
    grad_y = torch.tensor([0.0, 0.5, 0.75], dtype=torch.float32, device=device)
    for i in range(batch_size):
        train_y[i_beg:i_beg + 3, i, 0] = grad_y

    start_time = time.time()
    try:
        for epoch in range(train_epoch):
            with torch.no_grad():
                train_x = torch.randn((seq_len, batch_size, Inp_dim), dtype=torch.float32, device=device)

            # ten_mean = rd.uniform(0.5, 1.2, size=Inp_dim) * np.sign(rd.rand(Inp_dim) - 0.5)
            # ten_mean = torch.tensor(ten_mean, dtype=torch.float32, device=device)
            # ten_mean = ten_mean * mask01

            train_x = train_x + ten_mean

            inp = train_x
            out = net(inp)

            loss = (out - train_y) ** 2 * weights
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 128 == 0:  # 每 100 次输出结果
                print('Epoch: {:6}    Loss: {:.5f}'.format(epoch, loss.item()))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as error:
        print("Error:", error)
    finally:
        os.makedirs(Mod_dir, exist_ok=True)
        torch.save(net.state_dict(), '%s/actor.pth' % (Mod_dir,))
        print("Saved:", Mod_dir)
    print("Training Times Used:", int(time.time() - start_time))


def run_eval():
    eva_len = 5000
    eva_size = 1
    eva_tau = eva_len - 64

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(Inp_dim, Out_dim, Mid_dim, mid_layers=2).to(device)

    '''load model'''
    net.load_state_dict(torch.load('%s/actor.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

    eva_x = torch.randn((eva_len, eva_size, Inp_dim), dtype=torch.float32, device=device)
    eva_mask01 = torch.zeros((eva_len, eva_size, Inp_dim), dtype=torch.float32, device=device)
    eva_mask01[eva_tau:, :, :] = 1

    eva_mu = torch.tensor(Mu, dtype=torch.float32, device=device)
    eva_mu = eva_mu.view(1, 1, Inp_dim)
    eva_mean = eva_mu * torch.ones((eva_len, eva_size, Inp_dim), device=device)
    eva_inp = eva_x + eva_mean * eva_mask01

    eva_out = net(eva_inp)
    eva_out = eva_out.cpu().data.numpy()
    eva_out = eva_out.flatten()
    # np.save('temp.npy', eva_out)
    # print("Change Point: {}    Predict: {}".format(i_beg, ))
    draw_action_plot(eva_out, eva_tau)


def run_eval_loop():
    wait_t = 64
    eva_tau = 5000
    eva_len = eva_tau + wait_t
    eva_size = 1
    num_trials = 1000  # number of experiments

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(Inp_dim, Out_dim, Mid_dim, mid_layers=2).to(device)
    net.eval()

    '''load model'''
    net.load_state_dict(torch.load('%s/actor.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

    '''eval loop'''
    eva_mu = torch.tensor(Mu, dtype=torch.float32, device=device)
    eva_mu = eva_mu.view(1, 1, Inp_dim)
    eva_mean = eva_mu * torch.ones((eva_len, eva_size, Inp_dim), device=device)

    eva_mask01 = torch.zeros((eva_len, eva_size, Inp_dim), dtype=torch.float32, device=device)
    eva_mask01[eva_tau:, :, :] = 1

    for h in (0.3, 0.4):  # (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.99):
        c_false_alarm_events = 0
        c_false_alarm_periods = []
        c_det_delays = []

        for n in range(num_trials // eva_size):
            eva_x = torch.randn((eva_len, eva_size, Inp_dim),
                                dtype=torch.float32, device=device)

            eva_inp = eva_x + eva_mean * eva_mask01  # obs s

            eva_out = net(eva_inp)
            eva_out = eva_out.cpu().data.numpy()
            eva_out = eva_out.flatten()

            eva_out[-1] = 1.0
            # assert h <= 1.0
            eva_actions = np.where(eva_out >= h)[0]
            eva_t = eva_actions[0]

            det_delay = np.nan
            if eva_t < eva_tau:  # at this point, the online decision is 1 ("stop")
                c_false_alarm_events += 1
                c_false_alarm_periods.append(eva_t)
            else:
                det_delay = eva_t - eva_tau
                c_det_delays.append(det_delay)

            # for t in range(eva_len - 512):
            #     # if t < tau:
            #     #     obs = rd.randn(1, Inp_dim)  # pre-change observation
            #     # else:
            #     #     obs = rd.randn(1, Inp_dim) + Mu  # post-change observation
            #     eva_out = net(eva_inp[t:t + 512])
            #     # eva_out = eva_out.cpu().data.numpy()
            #     # eva_out = eva_out.flatten()
            #     eva_g = eva_out[-1, 0, 0].item()
            #     if eva_g >= h:
            #         break

            # det_delay = np.nan
            # if t < eva_tau:  # at this point, the online decision is 1 ("stop")
            #     c_false_alarm_events += 1
            #     c_false_alarm_periods.append(t)
            # else:
            #     det_delay = t - eva_tau
            #     c_det_delays.append(det_delay)

            # if n % 128 == 0:
            #     print("n: {:3}    FAR n: {:3}    ADD: {:.2f}".format(
            #         n, c_false_alarm_events, det_delay,
            #     ))

        c_avg_det_delay = np.mean(np.array(c_det_delays))
        c_false_alarm_rate = c_false_alarm_events / num_trials

        print("h: {:4}    CUSUM ADD: {:.3f}    FAR: {:.3f}".format(h, c_avg_det_delay, c_false_alarm_rate))


def draw_action_plot(eva_out, eva_tau):
    plt.ion()

    fig, axs = plt.subplots(2)

    ax0 = axs[0]
    eva_range = 8
    ax0.plot(eva_out[eva_tau - eva_range:eva_tau + eva_range], 'royalblue', label='pred')
    ax0.plot([eva_range, eva_range], [-1, +1], 'lightcoral', label='change point')
    ax0.set_facecolor('#f8f8f8')
    ax0.grid(color='white', linewidth=1.5)
    ax0.legend(loc='best')

    ax1 = axs[1]
    ax1.plot(eva_out, 'darkcyan', label='pred')
    ax1.plot([eva_tau, eva_tau], [-1, +1], 'lightcoral', label='change point')

    plt.savefig('{}/SignalDetectRNN.png'.format(Mod_dir))
    plt.pause(4)


if __name__ == '__main__':
    run_train()
    run_eval()
    # run_eval_loop()
