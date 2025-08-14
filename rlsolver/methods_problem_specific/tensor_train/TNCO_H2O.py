import os
import sys
import time
import torch as th
import torch.nn as nn
from tqdm import tqdm

from H2O_MISO import ObjectiveTask, OptimizerTask, OptimizerOpti
from H2O_MISO import opt_loop

from TNCO_env import TensorNetworkEnv  # get_nodes_list
from TNCO_env import NodesSycamoreN53M20, NodesSycamoreN12M14

TEN = th.Tensor

NodesList, BanEdges = NodesSycamoreN53M20, 0

WarmUpSize = 2 ** 13 if os.name != 'nt' else 2 ** 8
BufferSize = 2 ** 19
BufferRate = 0.25  # Buffer1Size = Buffer2Size * BufferCoff
BatchSize = 2 ** 11

NumRepeats = 16
IterMaxStep = 2 ** 12
IterMinLoss = 0.1
EmaGamma = 0.99
# NoiseRatio = 0.5  # n53m20, 12e4 second, 18.958
NoiseRatio = 1.2  # n53m20, 67e4 second, 18.851
Dims = (1024, 256, 512, 256, 512, 256)  # (1024, 512, 256, 512, 256)
HidDim = 2 ** 10
TrainSkip = 8

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

if os.name == 'nt':  # debug in WindowOS NewType
    IterMinLoss = 4
    # NodesList, BanEdges = NodesSycamoreN12M14, 0


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
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-2:]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class ObjModel(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(256, 256, 256)):
        super().__init__()
        assert out_dim > 0
        self.net = build_mlp(dims=[inp_dim, *dims, 8], activation=nn.ReLU, if_raw_out=True)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def get_output(self, x):
        return self.net(x)

    def forward(self, x):
        return self.net(x).mean(dim=1, keepdim=True)


class ReplayBuffer:  # for L2O
    def __init__(self, max_size: int, state_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.thetas = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
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

            self.thetas[p0:p1], self.thetas[0:p] = states[:p2], states[-p:]
            self.scores[p0:p1], self.scores[0:p] = scores[:p2], scores[-p:]
        else:
            self.thetas[self.p:p] = states
            self.scores[self.p:p] = scores
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.thetas[ids], self.scores[ids]

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_paths = (
            (self.thetas, f"{cwd}/buffer_thetas.pth"),
            (self.scores, f"{cwd}/buffer_scores.pth"),
        )

        if if_save:
            print(f"| buffer.save_or_load_history(): Save {cwd}    cur_size {self.cur_size}")
            for item, path in item_paths:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = th.vstack((item[self.p:self.cur_size], item[0:self.p]))
                th.save(buf_item.half(), path)  # save float32 as float16

        elif all([os.path.isfile(path) for item, path in item_paths]):
            max_sizes = []
            for item, path in item_paths:
                buf_item = th.load(path, map_location=th.device('cpu')).float()  # load float16 as float32
                print(f"| buffer.save_or_load_history(): Load {path}    {buf_item.shape}")

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

    def empty_and_reset_pointer(self):
        self.p = 0
        self.cur_size = 0
        self.if_full = False


def collect_buffer_history(if_remove: bool = False):
    max_size = 2 ** 18
    save_dir0 = 'task_TNCO'
    save_dirs = [save_dir for save_dir in os.listdir('') if save_dir[:9] == 'task_TNCO']

    states_ary = []
    scores_ary = []
    for save_dir in save_dirs:
        states_path = f"{save_dir}/buffer_thetas.pth"
        scores_path = f"{save_dir}/buffer_scores.pth"

        if_all_exists = all([os.path.isfile(path) for path in (states_path, scores_path)])
        if not if_all_exists:
            print(f"FileExist? [thetas, scores] {if_all_exists}")
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
    th.save(states_ary, f"{save_dir0}/buffer_thetas.pth")
    th.save(scores_ary, f"{save_dir0}/buffer_scores.pth")
    print(f"Save {save_dir0:12}    num_samples {scores_ary.shape[0]}")
    print(f"sort max {scores_ary[+0].item():9.3f}")
    print(f"sort min {scores_ary[-1].item():9.3f}")


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, num, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.num = num
        self.args = ()
        self.device = device

        self.num_eval = num

        self.env = TensorNetworkEnv(nodes_list=NodesList, ban_edges=BanEdges, device=device)
        self.dim = self.env.num_edges - self.env.ban_edges
        print(f"ObjectiveTNCO.dim {self.dim} != dim {dim}") if self.dim != dim else None

        self.obj_model1 = ObjModel(inp_dim=self.dim, out_dim=1, dims=Dims).to(device)
        self.obj_model0 = ObjModel(inp_dim=self.dim, out_dim=1, dims=Dims).to(device)
        self.obj_model1.train()
        self.obj_model0.eval()

        self.optimizer = th.optim.Adam(self.obj_model1.parameters(), lr=8e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = BatchSize
        self.ema_loss = IterMinLoss * 1.1
        self.ema_step = 0
        self.keep_score = -th.inf
        self.best_score = th.inf

        self.train_skip = TrainSkip
        self.train_count = 0

        gpu_id = -1 if self.device.index is None else self.device.index

        # 2023-05-22 18:03:14, todo, add historical node2s
        # build `self.historical_theta` before `self.random_generate_input_output`
        from TNCO_env import Node2sSycamoreN53N20COTE2
        node2s = Node2sSycamoreN53N20COTE2
        dim = self.dim
        if node2s is None:  # +arange
            theta = th.arange(dim, device=self.device).to(th.float32)
        else:
            edge_sort0 = th.arange(dim, device=device)
            edge_sort1 = self.env.convert_node2s_to_edge_sort(node2s=node2s, if_print=False).to(device)
            edge_sort2 = th.tensor([x for x in edge_sort0 if x not in edge_sort1], device=device, dtype = th.long)
            edge_sort = th.hstack((edge_sort1, edge_sort2))
            theta = th.zeros(dim, dtype=th.float32, device=device)
            theta[edge_sort] = th.arange(dim, device=device).to(th.float32)
        theta = (theta - theta.mean(dim=-1)) / (theta.std(dim=-1) + 1e-6)
        self.historical_theta = theta

        """init buffer"""
        self.save_path = f'./task_TNCO_{gpu_id:02}'
        os.makedirs(self.save_path, exist_ok=True)

        '''build buffer'''
        buf_max_size = BufferSize
        warm_up_size = WarmUpSize
        buf_rate0 = BufferRate
        buf_rate1 = 1 - BufferRate
        self.buffer1 = ReplayBuffer(max_size=int(buf_max_size * buf_rate1), state_dim=self.dim, gpu_id=gpu_id)
        self.buffer0 = ReplayBuffer(max_size=int(buf_max_size * buf_rate0), state_dim=self.dim, gpu_id=gpu_id)
        self.buffer1.save_or_load_history(cwd=self.save_path, if_save=False)

        if self.buffer1.cur_size < warm_up_size:  # warm_up
            noise_stds = (3.0, 2.0, 1.5, 1.2, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
            _warm_up_size = warm_up_size // len(noise_stds)
            for noise_std in noise_stds:
                thetas, scores = self.random_generate_input_output(num=_warm_up_size, noise_std=noise_std)
                self.buffer1.update(items=(thetas, scores))

        self.save_and_check_buffer()

        iter_max_step = IterMaxStep * 16
        while self.ema_loss > IterMinLoss:
            self.pbar = tqdm(total=iter_max_step, ascii=True)
            self.fast_train_obj_model(iter_max_step=iter_max_step)
            self.pbar.close()

    def get_objectives(self, thetas, *args) -> TEN:
        iter_max_step = IterMaxStep

        # shape == (warm_up_size, self.dim)
        thetas = self.get_norm(thetas, if_grad=True)
        with th.no_grad():
            scores = self.get_objectives_without_grad(thetas).unsqueeze(1)  # shape == (warm_up_size, 1)
            self.buffer1.update(items=(thetas, scores))

        self.train_count += 1
        if self.train_count == self.train_skip:
            self.train_count = 0
            self.fast_train_obj_model(iter_max_step=iter_max_step)
            self.hard_update(self.obj_model0, self.obj_model1)
        objectives = self.obj_model0(thetas)
        return objectives

    def get_objectives_without_grad(self, thetas, *_args) -> TEN:
        # assert theta.shape[0] == self.env.num_edges
        return self.env.get_log10_multiple_times(edge_sorts=thetas.argsort(dim=1))

    @staticmethod
    def get_norm(x_raw, if_grad=True):
        return (x_raw - x_raw.mean(dim=-1, keepdim=True)) / (x_raw.std(dim=-1, keepdim=True) + 1e-6)

    def random_generate_input_output(self, num: int = 512, noise_std=3.0):
        batch_size = 512

        theta0 = self.get_theta().unsqueeze(0)
        thetas = theta0 + th.randn((num, self.dim), dtype=th.float32, device=self.device) * noise_std
        thetas = self.get_norm(thetas, if_grad=False)

        scores = th.zeros((num, 1), dtype=th.float32, device=self.device)
        i_iter = tqdm(range(0, num, batch_size), ascii=True)
        for i in i_iter:
            i_iter.set_description(f"TNCO | warm_up {num}  noise {noise_std:.3f}")
            j = i + batch_size
            scores[i:j, 0] = self.get_objectives_without_grad(thetas[i:j])

        min_score = scores.min().item()
        avg_score = scores.mean().item()
        print(f"TNCO | {'':36}"
              f"min_score {min_score:7.3f}  "
              f"avg_score {avg_score:7.3f} ± {scores.std(dim=0).item():6.3f}"
              )
        return thetas, scores

    def fast_train_obj_model(self, iter_max_step: int):
        iter_min_loss = IterMinLoss

        ema_loss = iter_min_loss * 1.1  # Exponential Moving Average (EMA) loss value
        ema_gamma = EmaGamma

        ratio1 = self.buffer1.cur_size / (self.buffer1.cur_size + self.buffer0.cur_size)
        ratio0 = self.buffer0.cur_size / (self.buffer1.cur_size + self.buffer0.cur_size)

        batch_size1 = int(self.batch_size * ratio1)
        batch_size0 = int(self.batch_size * ratio0)

        pbar = self.pbar
        iter_step = 0
        for iter_step in range(1, iter_max_step + 1):
            inputs1, labels1 = self.buffer1.sample(batch_size1)
            inputs0, labels0 = self.buffer0.sample(batch_size0)

            inputs = th.vstack((inputs1, inputs0))
            labels = th.vstack((labels1, labels0))
            outputs = self.obj_model1.get_output(inputs)
            loss = self.criterion(outputs, labels.repeat(1, outputs.shape[1]))

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            ema_loss = ema_gamma * ema_loss + (1 - ema_gamma) * loss.item()

            pbar.n = iter_step
            pbar.set_description(f"Loss {ema_loss:9.3e}")
            pbar.update(0)

            if ema_loss < iter_min_loss:
                break

        self.ema_loss = ema_loss
        self.ema_step = iter_step

    def save_and_check_buffer(self):
        buffer_size = self.buffer1.cur_size + self.buffer0.cur_size

        buffer = ReplayBuffer(max_size=buffer_size, state_dim=self.dim, gpu_id=-1)
        buffer.update(items=(self.buffer1.get_cur_ten('thetas'), self.buffer1.get_cur_ten('scores')))
        buffer.update(items=(self.buffer0.get_cur_ten('thetas'), self.buffer0.get_cur_ten('scores')))
        buffer.save_or_load_history(cwd=self.save_path, if_save=True)

        '''move data to buffer1 buffer0'''
        thetas = buffer.get_cur_ten('thetas')
        scores = buffer.get_cur_ten('scores')
        del buffer

        self.keep_score = th.quantile(scores.squeeze(1), q=BufferRate).item()
        self.buffer1.empty_and_reset_pointer()
        self.buffer0.empty_and_reset_pointer()
        mask1 = scores.squeeze(1) > self.keep_score
        mask0 = ~mask1
        self.buffer1.update(items=(thetas[mask1], scores[mask1]))
        self.buffer0.update(items=(thetas[mask0], scores[mask0]))

        thetas0 = self.buffer0.get_cur_ten('thetas')
        scores0 = self.buffer0.get_cur_ten('scores')
        min_score = scores0.min().item()
        avg_score = scores0.mean().item()
        max_score = scores0.max().item()
        print(f"min {min_score:9.3f}    "
              f"max {max_score:9.3f}    "
              f"avg {avg_score:9.3f} ± {scores0.std(dim=0).item():9.3f}    "
              f"num {scores0.shape[0]}")

        if min_score < self.best_score:
            self.best_score = min_score
            print(f"best_result: {self.best_score:9.3f}    best_sort:\n"
                  f"{thetas0[scores0.argmin()].argsort().cpu().detach().numpy()}")

    @staticmethod
    def hard_update(target_net: th.nn.Module, current_net: th.nn.Module):
        for tar_param, cur_param in zip(target_net.parameters(), current_net.parameters()):
            tar_param.data.copy_(cur_param.data)

        for tar_module, cur_module in zip(target_net.modules(), current_net.modules()):
            if isinstance(cur_module, nn.BatchNorm1d) and isinstance(tar_module, nn.BatchNorm1d):
                tar_module.running_mean = cur_module.running_mean
                tar_module.running_var = cur_module.running_var

    def get_theta(self):  # load
        return self.historical_theta

    def get_thetas(self, num):
        theta0 = self.get_theta().unsqueeze(0)
        thetas = theta0 + th.randn((num, self.dim), dtype=th.float32, device=self.device) * NoiseRatio
        thetas = self.get_norm(thetas, if_grad=False)

        scores = self.buffer0.get_cur_ten('scores')
        ids = (scores < self.keep_score).nonzero(as_tuple=True)[0]
        num2 = num // 2
        rd_ids = th.randperm(len(ids))[:num2]
        thetas[:num2] = self.buffer0.thetas[ids[rd_ids]]
        return thetas

    def get_args_for_train(self):
        scores = self.buffer0.get_cur_ten('scores').squeeze(1)
        self.keep_score = th.quantile(scores, q=BufferRate).item()
        return ()

    def get_args_for_eval(self):
        return self.get_args_for_train()


def unit_test__objective_tnco():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    num = 3  # batch_size

    obj_task = ObjectiveTNCO(num=num, dim=-1, device=device)

    thetas = th.randn((obj_task.num, obj_task.dim), dtype=th.float32, device=obj_task.device)
    obj_task.get_objectives(thetas=thetas)
    obj_task.save_and_check_buffer()


"""trainable objective function"""


def train_optimizer():
    gpu_id = GPU_ID

    '''train'''
    train_times = 2 ** 10
    num = NumRepeats  # batch_size
    lr = 8e-4
    unroll = 16  # step of Hamilton Term
    num_opt = 128
    hid_dim = HidDim

    '''eval'''
    eval_gap = 2 ** 1
    import numpy as np
    np.set_printoptions(linewidth=120)

    '''init task'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    obj_task = ObjectiveTNCO(num=num, dim=-1, device=device)
    dim = obj_task.dim

    opt_task = OptimizerTask(num=num, dim=dim, device=device)
    opt_opti = OptimizerOpti(inp_dim=dim, hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    print('training start')
    start_time = time.time()
    for i in range(train_times + 1):
        obj_task.pbar = tqdm(total=IterMaxStep, ascii=True)

        for _ in range(eval_gap):
            _best_results, _min_losses = opt_loop(
                obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti, num_opt=num_opt,
                device=device, unroll=unroll, opt_base=opt_base, if_train=True)

        best_results, min_losses = opt_loop(
            obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti, num_opt=num_opt,
            device=device, unroll=unroll * 2, opt_base=opt_base, if_train=False)

        min_loss, ids = th.min(min_losses, dim=0)
        with th.no_grad():
            # best_result.shape == (1, -1)
            edge_sorts = best_results[ids].argsort()
            scores = obj_task.env.get_log10_multiple_times(edge_sorts=edge_sorts)
            score = scores.squeeze(0)

            time_used = time.time() - start_time
            obj_task.pbar.set_description(f"Loss {obj_task.ema_loss:9.3e}  Step {obj_task.ema_step:6}  "
                                          f"{i:>9}  {score:9.3f}  {min_loss.item():9.3e}  {time_used:9.3e} sec")
        obj_task.pbar.close()
        obj_task.save_and_check_buffer() if i % 4 == 0 else None
    print('training stop')


if __name__ == '__main__':
    train_optimizer()
    # unit_test__objective_tnco()
    # collect_buffer_history(if_remove=False)
    # random_search_from_arange()
