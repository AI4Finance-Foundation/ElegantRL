import os
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy

from TNCO_env import TensorNetworkEnv  # get_nodes_list
from L2O_H_term import ObjectiveTask, OptimizerTask, OptimizerOpti
from L2O_H_term import opt_train, opt_eval

from TNCO_env import \
    NodesSycamoreN53M12, \
    NodesSycamoreN53M14, \
    NodesSycamoreN53M16, \
    NodesSycamoreN53M18, \
    NodesSycamoreN53M20, \
    NodesSycamoreN12M14

TEN = th.Tensor

NodesList, BanEdges = NodesSycamoreN53M16, 0

WarmUpSize = 2 ** 14
BufferSize = 2 ** 19  # todo 18
BufferRate = 0.25  # Buffer1Size = Buffer2Size * BufferCoff
BatchSize = 2 ** 9  # todo 8

TrainGap = 8
NumRepeats = 16
NumPatience = 2 ** 9  # todo 10
EmaLossInit = 64
EmaGamma = 0.98

UnRoll = 16
NumOpt = 128
HidDimLSTM = 64

'''MLP + ReplayBuffer -> Trainable objective'''


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class ObjNet(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(128, 128 * 4)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, inp_dim), nn.Tanh(),
            nn.Linear(inp_dim, dims[0]), nn.Tanh(),
            DenseNet(lay_dim=dims[0]),
            nn.Linear(dims[1], out_dim),
        )
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = th.cat((x1, self.dense1(x1)), dim=1)
        return th.cat((x2, self.dense2(x2)), dim=1)  # x3  # x2.shape==(-1, lay_dim*4)


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
        self.scores = th.empty((max_size, 1), dtype=th.float32, device=self.device)

    def update(self, items: [TEN]):
        states, scores = items
        # assert thetas.shape == (warm_up_size, self.dim)
        # assert scores.shape == (warm_up_size, 1)

        p = self.p + scores.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.scores[p0:p1], self.scores[0:p] = scores[:p2], scores[-p:]
        else:
            self.states[self.p:p] = states
            self.scores[self.p:p] = scores
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.scores[ids]

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_names = (
            (self.states, "states"),
            (self.scores, "scores"),
        )

        if if_save:
            print(f"| buffer.save_or_load_history(): Save {cwd}    cur_size {self.cur_size}")
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = th.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                th.save(buf_item.half(), file_path)  # save float32 as float16

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                buf_item = th.load(file_path).float()  # load float16 as float32
                print(f"| buffer.save_or_load_history(): Load {file_path}    {buf_item.shape}")

                max_size = buf_item.shape[0]
                max_size = min(self.max_size, max_size)
                item[:max_size] = buf_item[-max_size:]  # load
                max_sizes.append(max_size)

            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]
            self.p = self.cur_size
            self.if_full = self.cur_size == self.max_size

    def get_cur_ten(self, var_name: str) -> TEN:
        ten = getattr(self, var_name)
        return ten[:self.cur_size]


def collect_buffer_history(if_remove: bool = False):
    max_size = BufferSize
    save_dir = f'task_TNCO'
    gpu_ids_list = [(0, 4), (1, 5), (2, 6), (3, 7)]
    type_names = ['m12', 'm14', 'm16', 'm20']

    for type_id, type_name in enumerate(type_names):
        gpu_ids = gpu_ids_list[type_id]

        for buf_type in ('buf0', 'buf1'):
            src_dirs = [f"{save_dir}_{gpu_id:02}_{buf_type}" for gpu_id in gpu_ids]

            states_ary = []
            scores_ary = []
            for src_dir in src_dirs:
                states_path = f"{src_dir}/replay_buffer_states.pth"
                scores_path = f"{src_dir}/replay_buffer_scores.pth"

                if_exists = [os.path.isfile(path) for path in (states_path, scores_path)]
                if not all(if_exists):
                    print(f"[states, scores]: {if_exists}    FileNotFound in: {src_dir}")
                    continue

                states = th.load(states_path, map_location=th.device('cpu')).half()
                scores = th.load(scores_path, map_location=th.device('cpu')).half()

                filter_ten = (th.isnan(scores) & th.isinf(scores)).squeeze(1)  # filter nan and inf
                filter_num = filter_ten.to(th.float32).sum()
                if filter_num != 0:
                    print(f"Exist nan or inf: {filter_num} in {src_dir}")
                states = states[~filter_ten]
                scores = scores[~filter_ten]

                states_ary.append(states)
                scores_ary.append(scores)

                os.remove(states_path) if if_remove else None
                os.remove(scores_path) if if_remove else None
                # print(f"Load {src_dir:12}    num_samples {scores.shape[0]}")

            states_ary = th.vstack(states_ary)
            scores_ary = th.vstack(scores_ary)

            sort = (-scores_ary).squeeze(1).argsort()[-max_size:]  # notice negative symbol here.
            states_ary = states_ary[sort]
            scores_ary = scores_ary[sort]

            '''save'''
            for src_dir in src_dirs:
                dst_dir = f"{save_dir}/{src_dir}"
                os.makedirs(dst_dir, exist_ok=True)
                th.save(states_ary, f"{dst_dir}/replay_buffer_states.pth")
                th.save(scores_ary, f"{dst_dir}/replay_buffer_scores.pth")

            scores = scores_ary.to(th.float64)
            min_score = scores[-1].item()  # = scores.min().item()
            avg_score = scores.mean().item()
            max_score = scores[+0].item()  # = scores.max().item()
            print(f"min_score: {min_score:9.3f}")
            print(f"avg_score: {avg_score:9.3f} ± {scores.std(dim=0).item():9.3f}")
            print(f"max_score: {max_score:9.3f}")
            print(f"type_names: {type_name}    states.shape: {list(states_ary.shape)}")


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.device = device
        self.args = ()

        self.env = TensorNetworkEnv(nodes_list=NodesList, ban_edges=BanEdges, device=device)
        self.dim = self.env.num_edges - self.env.ban_edges
        print(f"ObjectiveTNCO.dim: {self.dim} != dim: {dim}") if self.dim != dim else None

        self.obj_model = ObjNet(inp_dim=self.dim, out_dim=1).to(device)

        self.optimizer = th.optim.Adam(self.obj_model.parameters(), lr=4e-4)
        self.criterion = nn.MSELoss()
        self.best_loss = EmaLossInit
        self.ema_loss = EmaLossInit
        self.ema_counter = 0

        gpu_id = -1 if self.device.index is None else self.device.index

        """init buffer"""
        buf_rate0 = BufferRate
        buf_rate1 = 1 - BufferRate

        buf_max_size1 = int(BufferSize * buf_rate1)
        buf_max_size0 = int(BufferSize * buf_rate0)
        warm_up_size1 = int(WarmUpSize * buf_rate1)
        warm_up_size0 = int(WarmUpSize * buf_rate0)
        self.batch_size1 = int(BatchSize * buf_rate1)
        self.batch_size0 = int(BatchSize * buf_rate0)
        self.best_score = th.inf

        '''init buffer0 for best result'''
        self.save_path0 = f'./task_TNCO_{gpu_id:02}_buf0'
        os.makedirs(self.save_path0, exist_ok=True)
        self.buffer0 = ReplayBuffer(max_size=buf_max_size0, state_dim=self.dim, gpu_id=gpu_id)
        self.buffer0.save_or_load_history(cwd=self.save_path0, if_save=False)
        if self.buffer0.cur_size < warm_up_size0:
            thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size0, if_tqdm=True)
            self.buffer0.update(items=(thetas, scores))
            self.buffer0.save_or_load_history(cwd=self.save_path0, if_save=True)

        '''init buffer1 and warm up'''
        self.save_path1 = f'./task_TNCO_{gpu_id:02}_buf1'
        os.makedirs(self.save_path1, exist_ok=True)
        self.buffer1 = ReplayBuffer(max_size=buf_max_size1, state_dim=self.dim, gpu_id=gpu_id)
        self.buffer1.save_or_load_history(cwd=self.save_path1, if_save=False)
        if self.buffer1.cur_size < warm_up_size1:
            thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size1, if_tqdm=True)
            self.buffer1.update(items=(thetas, scores))
            self.buffer1.save_or_load_history(cwd=self.save_path0, if_save=True)

        self.pbar = tqdm(total=NumPatience, ascii=True)
        self.obj_model_path = f"{self.save_path0}/obj_model.pth"
        if os.path.isfile(self.obj_model_path):  # load obj_model before self.fast_train_obj_model
            self.obj_model = th.load(self.obj_model_path, map_location=device)
        self.fast_train_obj_model()
        self.pbar.close()

        self.train_counter = 0
        self.buffer0_thresh = -th.inf

    def get_objective(self, theta, *args) -> TEN:
        num_repeats = NumRepeats

        with th.no_grad():
            thetas = theta.repeat(num_repeats, 1)
            thetas[1:] += th.rand_like(thetas[1:])
            thetas: TEN = self.get_norm(thetas)  # shape == (warm_up_size, self.dim)
            scores: TEN = self.get_objectives_without_grad(thetas).unsqueeze(1)  # shape == (warm_up_size, 1)

            indices = scores.squeeze(1) < self.buffer0_thresh
            self.buffer0.update(items=(thetas[indices], scores[indices]))
            self.buffer1.update(items=(thetas[~indices], scores[~indices]))

        self.train_counter += 1
        if self.train_counter > TrainGap:
            self.train_counter = 0
            self.fast_train_obj_model()

            buffer0_scores = self.buffer0.get_cur_ten('scores')[::64]
            self.buffer0_thresh = th.quantile(buffer0_scores, q=BufferRate).item()

        obj_model1 = deepcopy(self.obj_model)
        # avoid `gradient computation has been modified by an inplace operation` inv `all_losses.backward()`
        objective = obj_model1(theta.unsqueeze(0)).squeeze(0)
        return objective

    def get_objectives_without_grad(self, thetas, *_args) -> TEN:
        # assert theta.shape[0] == self.env.num_edges
        with th.no_grad():
            log10_multiple_times = self.env.get_log10_multiple_times(edge_sorts=thetas.argsort(dim=1))
        return log10_multiple_times

    @staticmethod
    def get_norm(x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def random_generate_input_output(self, warm_up_size: int = 512, batch_size: int = 1024, if_tqdm: bool = False):
        print(f"TNCO | random_generate_input_output: num_warm_up={warm_up_size}")
        thetas = th.randn((warm_up_size, self.dim), dtype=th.float32, device=self.device)
        thetas = ((thetas - thetas.mean(dim=1, keepdim=True)) / (thetas.std(dim=1, keepdim=True) + 1e-6))

        batch_size = batch_size if batch_size <= warm_up_size else warm_up_size
        thetas_iter = thetas.reshape((-1, batch_size, self.dim))
        if if_tqdm:
            from tqdm import tqdm
            thetas_iter = tqdm(thetas_iter, ascii=True)
        scores = th.hstack([self.get_objectives_without_grad(thetas) for thetas in thetas_iter]).unsqueeze(1)

        # assert thetas.shape == (warm_up_size, self.dim)
        # assert scores.shape == (warm_up_size, 1)
        return thetas, scores

    def fast_train_obj_model(self):
        best_loss = EmaLossInit
        ema_loss = EmaLossInit  # Exponential Moving Average (EMA) loss value
        ema_gamma = EmaGamma

        num_patience = NumPatience
        num_iter = 0
        num_counter = 0
        pbar = self.pbar

        while num_iter < num_patience:
            inputs1, labels1 = self.buffer1.sample(self.batch_size1)
            inputs0, labels0 = self.buffer0.sample(self.batch_size0)
            inputs = th.vstack((inputs1, inputs0))
            labels = th.vstack((labels1, labels0))

            outputs = self.obj_model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            ema_loss = ema_gamma * ema_loss + (1 - ema_gamma) * loss.item()

            num_counter += 1
            num_iter += 1
            if best_loss > ema_loss:
                if num_iter > num_patience * 0.1:
                    pbar.n = num_iter
                pbar.set_description(f"EmaLoss {ema_loss:9.3e}  Counter {num_counter:9}")
                pbar.update(0)

                best_loss = ema_loss
                num_iter = 0
        pbar.set_description(f"EmaLoss {ema_loss:9.3e}  Counter {num_counter:9}")
        pbar.n = num_patience
        pbar.update(0)

        self.ema_loss = ema_loss
        self.ema_counter = num_counter

    def save_and_check_buffer(self):
        self.buffer0.save_or_load_history(cwd=self.save_path0, if_save=True)
        self.buffer1.save_or_load_history(cwd=self.save_path1, if_save=True)

        scores = self.buffer0.scores[:self.buffer0.cur_size]
        min_score = scores.min().item()
        avg_score = scores.mean().item()
        max_score = scores.max().item()
        print(f"num_train: {scores.shape[0]}")
        print(f"min_score: {min_score:9.3f}")
        print(f"avg_score: {avg_score:9.3f} ± {scores.std(dim=0).item():9.3f}")
        print(f"max_score: {max_score:9.3f}")

        if min_score < self.best_score:
            th.save(self.obj_model, self.obj_model_path)

            self.best_score = min_score
            print(f"best_result: {self.best_score:9.3f}    best_sort:\n"
                  f"{self.buffer0.states[scores.argmin()].argsort().cpu().detach().numpy()}")


def unit_test__objective():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = ObjectiveTNCO(dim=0, device=device)
    obj_task.get_objective(theta=th.rand(obj_task.dim, dtype=th.float32, device=obj_task.device))
    obj_task.save_and_check_buffer()


"""H2O: Hamilton Term + Learn to Optimize via Trainable objective"""


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 12
    unroll = UnRoll
    num_opt = NumOpt
    hid_dim = HidDimLSTM

    '''eval'''
    eval_gap = 2 ** 2
    np.set_printoptions(linewidth=128)

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    dim = 414  # set by env.num_edges
    obj_task = ObjectiveTNCO(dim=dim, device=device)
    dim = obj_task.env.num_edges

    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerOpti(hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=4e-4)

    '''loop'''
    start_time = time.time()
    for i in range(train_times + 1):
        '''train'''
        obj_task.pbar = tqdm(total=NumPatience, ascii=True)
        for _ in range(eval_gap):
            opt_train(obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                      num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        '''eval'''
        min_result, min_loss = opt_eval(obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                                        num_opt=num_opt * 2, device=device)
        with th.no_grad():
            edge_sorts = min_result.argsort().unsqueeze(0)
            scores = obj_task.env.get_log10_multiple_times(edge_sorts=edge_sorts)
            score = scores.squeeze(0)
            time_used = time.time() - start_time
            obj_task.pbar.set_description(f"EmaLoss {obj_task.ema_loss:9.3e}  Counter {obj_task.ema_counter:6}  "
                                          f"{i:>9}  {score:9.3f}  {min_loss.item():9.3e}  {time_used:9.3e} sec")
        obj_task.pbar.close()

        '''save'''
        if i % 4 == 0:
            obj_task.save_and_check_buffer()

    obj_task.save_and_check_buffer()


if __name__ == '__main__':
    # unit_test__objective_tnco()
    train_optimizer()
    # collect_buffer_history(if_remove=False)

"""
min_score:    15.531
avg_score:    24.387 ±     4.990
max_score:    49.375
type_names: m12    states.shape: [171814, 414]

min_score:    16.500
avg_score:    24.874 ±     6.104
max_score:    54.500
type_names: m14    states.shape: [175331, 484]

min_score:    20.219
avg_score:    30.698 ±     9.330
max_score:    67.438
type_names: m16    states.shape: [123673, 585]

min_score:    22.734
avg_score:    32.146 ±     9.677
max_score:    81.000
type_names: m20    states.shape: [254913, 754]
"""
