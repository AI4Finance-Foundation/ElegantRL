import os
import sys
import time
import torch as th
import torch.nn as nn
from copy import deepcopy

from env_g49 import TensorNetworkEnv  # get_nodes_list
from L2O_H_term import ObjectiveTask, OptimizerTask, OptimizerOpti
from L2O_H_term import opt_train, opt_eval

from env_g49 import \
    NodesSycamoreN53M12, \
    NodesSycamoreN53M14, \
    NodesSycamoreN53M16, \
    NodesSycamoreN53M18, \
    NodesSycamoreN53M20

TEN = th.Tensor


NodesList, BanEdges = NodesSycamoreN53M18, 0

WarmUpSize = 2 ** 14
NumRepeats = 4
EmaLossInit = 64
TrainThresh = 2 ** -5
EmaGamma = 0.98
MaxEpoch = 2 ** 9


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


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(256, 256, 256)):
        super().__init__()
        self.net = build_mlp(dims=[inp_dim, *dims, out_dim], activation=nn.Tanh)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, x):
        return self.net(x)


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
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = th.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}    {buf_item.shape}")
                th.save(buf_item.half(), file_path)  # save float32 as float16

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                buf_item = th.load(file_path).float()  # load float16 as float32
                print(f"| buffer.save_or_load_history(): Load {file_path}    {buf_item.shape}")

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]
            self.if_full = self.cur_size == self.max_size


def collect_buffer_history(if_remove: bool = False):
    max_size = 2 ** 18
    save_dir0 = 'task_TNCO'
    save_dirs = [save_dir for save_dir in os.listdir('') if save_dir[:9] == 'task_TNCO']

    states_ary = []
    scores_ary = []
    for save_dir in save_dirs:
        states_path = f"{save_dir}/replay_buffer_states.pth"
        scores_path = f"{save_dir}/replay_buffer_scores.pth"

        if_all_exists = all([os.path.isfile(path) for path in (states_path, scores_path)])
        if not if_all_exists:
            print(f"FileExist? [states, scores] {if_all_exists}")
            continue

        states = th.load(states_path, map_location=th.device('cpu')).half()
        scores = th.load(scores_path, map_location=th.device('cpu')).half()
        states_ary.append(states)
        scores_ary.append(scores)

        os.remove(states_path) if if_remove else None
        os.remove(scores_path) if if_remove else None
        print(f"Load {save_dir:12}    num_samples {scores.shape[0]}")

    states_ary = th.vstack(states_ary)
    scores_ary = th.vstack(scores_ary)

    sort = -scores_ary.squeeze(1).argsort()[:max_size]  # notice negative symbol here.
    states_ary = states_ary[sort]
    scores_ary = scores_ary[sort]

    os.makedirs(save_dir0, exist_ok=True)
    th.save(states_ary, f"{save_dir0}/replay_buffer_states.pth")
    th.save(scores_ary, f"{save_dir0}/replay_buffer_scores.pth")
    print(f"Save {save_dir0:12}    num_samples {scores_ary.shape[0]}")
    print(f"sort max {scores_ary[+0].item():9.3f}")
    print(f"sort min {scores_ary[-1].item():9.3f}")


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.device = device
        self.args = ()

        self.env = TensorNetworkEnv(nodes_list=NodesList, ban_edges=BanEdges, device=device)
        self.dim = self.env.num_edges - self.env.ban_edges
        print(f"ObjectiveTNCO.dim {self.dim} != dim {dim}") if self.dim != dim else None

        self.obj_model = MLP(inp_dim=self.dim, out_dim=1, dims=(256, 256, 256)).to(device)

        self.optimizer = th.optim.Adam(self.obj_model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = 2 ** 10
        self.train_thresh = TrainThresh
        self.ema_loss = 0.0

        gpu_id = -1 if self.device.index is None else self.device.index
        self.save_path = f'./task_TNCO_{gpu_id:02}'
        os.makedirs(self.save_path, exist_ok=True)

        '''warm up'''
        warm_up_size = WarmUpSize
        self.buffer = ReplayBuffer(max_size=2 ** 18, state_dim=self.dim, gpu_id=gpu_id)
        self.buffer.save_or_load_history(cwd=self.save_path, if_save=False)
        if self.buffer.cur_size < warm_up_size:
            thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size, if_tqdm=True)
            self.buffer.update(items=(thetas, scores))
        self.save_and_check_buffer()
        self.fast_train_obj_model()

        # thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size, if_tqdm=True)
        # self.buffer.update(items=(thetas, scores))
        # self.save_and_check_buffer()
        # self.fast_train_obj_model()
        # exit()

    def get_objective(self, theta, *args) -> TEN:
        num_repeats = NumRepeats
        with th.no_grad():
            thetas = theta.repeat(num_repeats, 1)
            thetas[1:] += th.rand_like(thetas[1:])
            thetas = self.get_norm(thetas)  # shape == (warm_up_size, self.dim)
            scores = self.get_objectives_without_grad(thetas).unsqueeze(1)  # shape == (warm_up_size, 1)
            self.buffer.update(items=(thetas, scores))

        self.fast_train_obj_model()

        obj_model1 = deepcopy(self.obj_model)
        # avoid `gradient computation has been modified by an inplace operation` in `all_losses.backward()` !!!

        objective = obj_model1(theta)
        return objective

    def get_objectives_without_grad(self, thetas, *_args) -> TEN:
        # assert theta.shape[0] == self.env.num_edges
        with th.no_grad():
            log10_multiple_times = self.env.get_log10_multiple_times(edge_sorts=thetas.argsort(dim=1))
        return log10_multiple_times

    @staticmethod
    def get_norm(x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def random_generate_input_output(self, warm_up_size: int = 512, if_tqdm: bool = False):
        print(f"TNCO | random_generate_input_output: num_warm_up={warm_up_size}")
        thetas = th.randn((warm_up_size, self.dim), dtype=th.float32, device=self.device)
        thetas = ((thetas - thetas.mean(dim=1, keepdim=True)) / (thetas.std(dim=1, keepdim=True) + 1e-6))

        thetas_iter = thetas.reshape((-1, 512, self.dim))
        if if_tqdm:
            from tqdm import tqdm
            thetas_iter = tqdm(thetas_iter, ascii=True)
        scores = th.hstack([self.get_objectives_without_grad(thetas) for thetas in thetas_iter]).unsqueeze(1)

        # assert thetas.shape == (warm_up_size, self.dim)
        # assert scores.shape == (warm_up_size, 1)
        return thetas, scores

    def fast_train_obj_model(self):
        train_thresh = self.train_thresh

        ema_loss = EmaLossInit  # Exponential Moving Average (EMA) loss value
        ema_gamma = EmaGamma
        max_epoch = MaxEpoch

        for counter in range(max_epoch):
            inputs, labels = self.buffer.sample(self.batch_size)

            outputs = self.obj_model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            ema_loss = ema_gamma * ema_loss + (1 - ema_gamma) * loss.item()
            if ema_loss < train_thresh:
                break

        self.ema_loss = ema_loss
        # print(f"     counter {counter:9}    ema_loss {ema_loss:9.2f}")

    def save_and_check_buffer(self):
        self.buffer.save_or_load_history(cwd=self.save_path, if_save=True)

        scores = self.buffer.scores[:self.buffer.cur_size]
        print(f"num_train: {scores.shape[0]}")
        print(f"min_score: {scores.min().item():9.3f}")
        print(f"avg_score: {scores.mean().item():9.3f} ± {scores.std(dim=0).item():9.3f}")
        print(f"max_score: {scores.max().item():9.3f}")
        print(f"best_result: \n{self.buffer.states[scores.argmin()].argsort()}")


def unit_test__objective_tnco():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = ObjectiveTNCO(dim=0, device=device)
    obj_task.get_objective(theta=th.rand(obj_task.dim, dtype=th.float32, device=obj_task.device))
    obj_task.save_and_check_buffer()


"""trainable objective function"""


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 12
    lr = 2e-4
    unroll = 16
    num_opt = 128
    hid_dim = 64

    '''eval'''
    eval_gap = 2 ** 2

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    dim = 414  # set by env.num_edges
    obj_task = ObjectiveTNCO(dim=dim, device=device)
    dim = obj_task.env.num_edges

    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerOpti(hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    start_time = time.time()
    '''loop'''
    for i in range(train_times + 1):
        opt_train(obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 0:
            best_result, min_loss = opt_eval(obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                                             num_opt=num_opt * 2, device=device)

            '''get score'''
            with th.no_grad():
                edge_sorts = best_result.argsort().unsqueeze(0)
                scores = obj_task.env.get_log10_multiple_times(edge_sorts=edge_sorts)
                score = scores.squeeze(0)

            time_used = time.time() - start_time
            print(f"{i:>9}    {score:9.3f}    {min_loss.item():9.3e}    TimeUsed {time_used:9.0f}")

        if i % (eval_gap * 4) == 0:
            obj_task.save_and_check_buffer()

    obj_task.save_and_check_buffer()


if __name__ == '__main__':
    # unit_test__objective_tnco()
    train_optimizer()
    # collect_buffer_history(if_remove=False)
