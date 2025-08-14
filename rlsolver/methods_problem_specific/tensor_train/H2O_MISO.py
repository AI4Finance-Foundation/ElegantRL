import sys
import time
import torch as th
import torch.nn as nn
from functorch import vmap

TEN = th.Tensor

"""Learn To Optimize + Hamilton Term

想要使用 pytorch 的 functorch.vmap，就要先安装 functorch

1. 先安装 conda

2. 打开终端并输入以下命令以添加阿里云的镜像源：
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/

3. 然后，输入以下命令创建一个新的 conda 环境并安装 functorch：
conda create --name myenv
conda activate myenv
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install functorch

4. 最后，运行以下命令以确认是否安装成功：
python -c "import functorch; print(functorch.__version__)"
"""


class ObjectiveTask:
    def __init__(self, *args):
        self.num = None
        self.num_eval = None
        self.dims = None
        self.args = ()

    def get_args_for_train(self):
        return self.args

    def get_args_for_eval(self):
        return self.args

    @staticmethod
    def get_objectives(*_args) -> TEN:
        return th.zeros()

    @staticmethod
    def get_norm(x):
        return x

    def get_thetas(self, num: int):
        return None


class OptimizerTask(nn.Module):
    def __init__(self, num, dim, device, thetas=None):
        super().__init__()
        self.num = num
        self.dim = dim
        self.device = device

        with th.no_grad():
            if thetas is None:
                thetas = th.randn((self.num, self.dim), requires_grad=True, device=device)
                thetas = (thetas - thetas.mean(dim=-1, keepdim=True)) / (thetas.std(dim=-1, keepdim=True) + 1e-6)
                thetas = thetas.clamp(-3, +3)
            else:
                thetas = thetas.clone().detach()
                assert thetas.shape[0] == num
        self.register_buffer('thetas', thetas.requires_grad_(True))

    def re_init(self, num, thetas=None):
        self.__init__(num=num, dim=self.dim, device=self.device, thetas=thetas)

    def get_outputs(self):
        return self.thetas


class OptimizerOpti(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.num_rnn = 2

        self.activation = nn.Tanh()
        self.recurs1 = nn.GRUCell(inp_dim, hid_dim)
        self.recurs2 = nn.GRUCell(hid_dim, hid_dim)
        self.output0 = nn.Linear(hid_dim * self.num_rnn, 1)
        self.output1 = nn.Linear(hid_dim * self.num_rnn, inp_dim)
        layer_init_with_orthogonal(self.output0, std=0.1)

    def forward(self, inp0, hid_):
        hid1 = self.activation(self.recurs1(inp0, hid_[0]))
        hid2 = self.activation(self.recurs2(hid1, hid_[1]))

        hid = th.cat((hid1, hid2), dim=1)
        out_avg = self.output0(hid)
        out_res = self.output1(hid)
        out = out_avg + out_res
        return out, (hid1, hid2)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


def opt_loop(
        obj_task: ObjectiveTask,
        opt_opti: OptimizerOpti,
        opt_task: OptimizerTask,
        opt_base: th.optim,
        num_opt: int,
        unroll: int,
        device: th.device,
        if_train: bool = True,
):
    if if_train:
        opt_opti.train()
        obj_args = obj_task.get_args_for_train()
        num = obj_task.num
    else:
        opt_opti.eval()
        obj_args = obj_task.get_args_for_eval()
        num = obj_task.num_eval

    thetas = obj_task.get_thetas(num=num)
    opt_task.re_init(num=num, thetas=thetas)

    opt_task.zero_grad()

    hid_dim = opt_opti.hid_dim
    hid_state1 = [th.zeros((num, hid_dim), device=device) for _ in range(opt_opti.num_rnn)]

    outputs_list = []
    losses_list = []
    all_losses = []

    th.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        outputs = opt_task.get_outputs()
        outputs = obj_task.get_norm(outputs)

        losses = obj_task.get_objectives(outputs, *obj_args)
        loss = losses.mean()
        loss.backward(retain_graph=True)

        all_losses.append(losses)

        '''record for selecting best output'''
        outputs_list.append(outputs.clone())
        losses_list.append(losses.clone())

        '''params update with gradient'''
        thetas = opt_task.thetas
        gradients = thetas.grad.detach().clone().requires_grad_(True)

        updates, hid_states2 = opt_opti(gradients, hid_state1)

        result = thetas + updates
        result = obj_task.get_norm(result)
        result.retain_grad()
        result_params = {'thetas': result}

        if if_train:
            if iteration % unroll == 0:
                # all_loss = th.min(th.stack(all_losses[iteration - unroll:iteration]), dim=0)[0].mean()
                all_loss = th.stack(all_losses[iteration - unroll:iteration]).mean()
                opt_base.zero_grad()
                all_loss.backward()
                opt_base.step()

                opt_task.re_init(num=num)
                opt_task.load_state_dict(result_params)
                opt_task.zero_grad()

                hid_state1 = [ten.detach().clone().requires_grad_(True) for ten in hid_states2]
            else:
                opt_task.thetas = result_params['thetas']

                hid_state1 = hid_states2
        else:
            opt_task.re_init(num=num)
            opt_task.load_state_dict(result_params)
            opt_task.zero_grad()

            hid_state1 = [ten.detach().clone().requires_grad_(True) for ten in hid_states2]

    th.set_grad_enabled(False)

    '''record for selecting best output'''
    losses_list = th.stack(losses_list)
    min_losses, ids = th.min(losses_list, dim=0)

    outputs_list = th.stack(outputs_list)
    best_outputs = outputs_list[ids.squeeze(1), th.arange(num, device=device)]

    return best_outputs, min_losses


"""run"""


class ObjectiveMISO(ObjectiveTask):
    def __init__(self, num, dims, device):
        super(ObjectiveMISO, self).__init__()
        self.num = num
        self.dim = th.prod(th.tensor(dims)).item()
        self.args = ()
        self.device = device

        self.dims = dims

        h_evals = self.load_from_disk(device)
        self.num_eval = h_evals.shape[0]
        self.h_evals = h_evals
        self.p_evals = (1, 10, 100)

        self.get_results_of_mmse = vmap(self.get_result_of_mmse, in_dims=(0, None), out_dims=0)
        self.get_objective_vmap = vmap(self.get_objective, in_dims=(0, 0), out_dims=0)

        loss_mmse_list = []
        for p_eval in self.p_evals:
            p_eval = th.tensor(p_eval, device=device)
            ws_mmse = self.get_results_of_mmse(h_evals, p_eval)

            h_scales = h_evals * (p_eval ** 0.5)
            loss_mmse = -self.get_objectives(ws_mmse, h_scales)
            loss_mmse_list.append(loss_mmse.mean().item())
        print(f"{'MMSE':>8} {loss_mmse_list[0]:>9.3f} {loss_mmse_list[1]:>9.3f} {loss_mmse_list[2]:>9.3f}\n")

    def get_args_for_train(self):
        p = 10 ** (th.rand(1).item() + 1)
        h_scales = (p ** 0.5) * th.randn((self.num, *self.dims), dtype=th.float32, device=self.device)
        args = (h_scales,)
        return args

    def get_args_for_eval(self):
        return self.args

    @staticmethod
    def get_objective(w: TEN, h: TEN, noise: float = 1.) -> TEN:
        w = w[0] + 1j * w[1]
        h = h[0] + 1j * h[1]

        hw = h @ w
        abs_hw_squared = th.abs(hw) ** 2
        signal = th.diagonal(abs_hw_squared)
        interference = abs_hw_squared.sum(dim=-1) - signal
        sinr = signal / (interference + noise)
        return -th.log2(1 + sinr).sum()

    def get_objectives(self, thetas: TEN, hs: TEN) -> TEN:
        ws = thetas.reshape(-1, *self.dims)
        return self.get_objective_vmap(ws, hs)

    @staticmethod
    def get_norm(x):
        return x / x.norm(dim=1, keepdim=True)

    @staticmethod
    def load_from_disk(device):
        import pickle
        with open(f'./K8N8Samples=100.pkl', 'rb') as f:
            h_evals = th.as_tensor(pickle.load(f), dtype=th.cfloat, device=device)
            assert h_evals.shape == (100, 8, 8)

        h_evals = th.stack((h_evals.real, h_evals.imag), dim=1)
        assert h_evals.shape == (100, 2, 8, 8)
        return h_evals

    @staticmethod
    def get_result_of_mmse(h, p) -> TEN:  # MMSE beamformer
        h = h[0] + 1j * h[1]
        k, n = h.shape
        eye_mat = th.eye(n, dtype=h.dtype, device=h.device)
        w = th.linalg.solve(eye_mat * k / p + th.conj(th.transpose(h, 0, 1)) @ h, th.conj(th.transpose(h, 0, 1)))
        w = w / (th.norm(w, dim=0, keepdim=True) * k ** 0.5)  # w.shape == [K, N]
        return th.stack((w.real, w.imag), dim=0)  # return.shape == [2, K, N]


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 10
    num = 2 ** 10  # batch_size
    lr = 8e-4
    unroll = 16  # step of Hamilton Term
    num_opt = 256
    hid_dim = 2 ** 7

    '''eval'''
    eval_gap = 2 ** 7

    '''init task'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    dims = (2, 8, 8)
    obj_task = ObjectiveMISO(num=num, dims=dims, device=device)
    dim = obj_task.dim

    '''init opti'''
    opt_task = OptimizerTask(num=num, dim=dim, device=device)
    opt_opti = OptimizerOpti(inp_dim=dim, hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    print('training start')
    start_time = time.time()
    for i in range(train_times + 1):
        opt_loop(
            obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
            num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base, if_train=True)

        if i % eval_gap == 1:
            loss_of_p_evals = []
            for p_eval in obj_task.p_evals:
                h_evals = obj_task.h_evals
                h_scales = h_evals * (p_eval ** 0.5)
                obj_task.args = (h_scales,)
                best_results, min_losses = opt_loop(
                    obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                    num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base, if_train=False)
                if i == train_times:
                    min_losses = obj_task.get_objectives(best_results, h_scales)
                loss_of_p_evals.append(-min_losses.mean().item())

            time_used = round((time.time() - start_time))
            print(f"{'H2O':>8} {loss_of_p_evals[0]:>9.3f} {loss_of_p_evals[1]:>9.3f} {loss_of_p_evals[2]:>9.3f}    "
                  f"TimeUsed {time_used:9}")
    print('training stop')


if __name__ == '__main__':
    
    train_optimizer()
