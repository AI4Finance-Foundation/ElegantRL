import sys
import time
import torch as th
import torch.nn as nn

TEN = th.Tensor

"""Learn To Optimize + Hamilton Term"""


class ObjectiveTask:
    def __init__(self, *args):
        self.args = None

    def get_args_for_train(self):
        return self.args

    def get_args_for_eval(self):
        return self.args

    @staticmethod
    def get_objective(*args) -> TEN:
        return th.zeros()

    @staticmethod
    def get_norm(x):
        return x


class OptimizerTask(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

        with th.no_grad():
            theta = th.rand(self.dim, requires_grad=True, device=device)
            theta = (theta - theta.mean(dim=-1, keepdim=True)) / (theta.std(dim=-1, keepdim=True) + 1e-6)
            theta = theta.clamp(-3, +3)  # todo
        self.register_buffer('theta', theta.requires_grad_(True))

    def re_init(self):
        self.__init__(dim=self.dim, device=self.device)

    def get_register_params(self):
        return [('theta', self.theta)]

    def get_output(self):
        return self.theta


class OptimizerOpti(nn.Module):
    def __init__(self, hid_dim=20):
        super().__init__()
        self.hid_dim = hid_dim
        self.recurs1 = nn.LSTMCell(1, hid_dim)
        self.recurs2 = nn.LSTMCell(hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, 1)

    def forward(self, inp0, hid0, cell):
        hid1, cell1 = self.recurs1(inp0, (hid0[0], cell[0]))
        hid2, cell2 = self.recurs2(hid1, (hid0[1], cell[1]))
        return self.output(hid2), (hid1, hid2), (cell1, cell2)


def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


def opt_train(
        obj_task: ObjectiveTask,
        opt_opti: OptimizerOpti,
        opt_task: OptimizerTask,
        opt_base: th.optim,
        num_opt: int,
        unroll: int,
        device: th.device,
):
    opt_opti.train()
    opt_task.zero_grad()

    obj_args = obj_task.get_args_for_train()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += th.tensor(p.shape).prod().item()
    hc_state1 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

    all_losses_ever = []
    all_losses = None

    th.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        output = opt_task.get_output()
        output = obj_task.get_norm(output)
        loss = obj_task.get_objective(output, *obj_args)
        loss.backward(retain_graph=True)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        all_losses_ever.append(loss.data.cpu().numpy())

        i = 0
        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)
        for name, p in opt_task.get_register_params():
            hid_dim = th.tensor(p.shape).prod().item()
            gradients = p.grad.view(hid_dim, 1).detach().clone().requires_grad_(True)

            j = i + hid_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_opti(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = obj_task.get_norm(result)
            result_params[name].retain_grad()

            i = j

        if iteration % unroll == 0:
            opt_base.zero_grad()
            all_losses.backward()
            opt_base.step()

            all_losses = None

            opt_task.re_init()
            opt_task.load_state_dict(result_params)
            opt_task.zero_grad()

            hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        else:
            for name, p in opt_task.get_register_params():
                set_attr(opt_task, name, result_params[name])

            hc_state1 = hc_state2
    th.set_grad_enabled(False)
    return all_losses_ever


def opt_eval(
        obj_task: ObjectiveTask,
        opt_opti: OptimizerOpti,
        opt_task: OptimizerTask,
        num_opt: int,
        device: th.device
):
    opt_opti.eval()

    obj_args = obj_task.get_args_for_eval()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += th.tensor(p.shape).prod().item()
    hc_state1 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

    loss = None
    best_res = None
    min_loss = th.inf

    th.set_grad_enabled(True)
    for _ in range(num_opt):
        output = opt_task.get_output()
        output = obj_task.get_norm(output)
        loss = obj_task.get_objective(output, *obj_args)
        loss.backward(retain_graph=True)

        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

        i = 0
        for name, p in opt_task.get_register_params():
            param_dim = th.tensor(p.shape).prod().item()
            gradients = p.grad.view(param_dim, 1).detach().clone().requires_grad_(True)

            j = i + param_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_opti(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = obj_task.get_norm(result)

            i = j

        opt_task.re_init()
        opt_task.load_state_dict(result_params)
        opt_task.zero_grad()

        hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        if loss < min_loss:
            best_res = opt_task.get_output()
            best_res = obj_task.get_norm(best_res)
            min_loss = loss
    assert not th.isnan(loss)
    th.set_grad_enabled(False)
    return best_res, min_loss


"""run"""


class ObjectiveMISO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveMISO, self).__init__()
        self.dim = dim
        self.device = device

        self.args = None

        h_evals = self.load_from_disk(device)
        self.h_evals = h_evals
        self.p_evals = [1, 10, 100]

        loss_mmse_list = []
        for p_eval in self.p_evals:
            loss_mmse = []
            for h in h_evals:
                h_scale = h * (p_eval ** 0.5)
                w_mmse = self.get_result_of_compare_method(h, p_eval)
                loss_mmse.append(-self.get_objective(w_mmse, h_scale).item())
            loss_mmse = sum(loss_mmse) / len(loss_mmse)
            loss_mmse_list.append(loss_mmse)
        print(f"{'MMSE':>8} {loss_mmse_list[0]:>9.3f} {loss_mmse_list[1]:>9.3f} {loss_mmse_list[2]:>9.3f}")

    def get_args_for_train(self):
        p = 10 ** (th.rand(1).item() + 1)
        h_scale = (p ** 0.5) * th.randn(self.dim, dtype=th.float32, device=self.device)
        args = (h_scale,)
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

    @staticmethod
    def get_norm(x):
        return x / x.norm()

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
    def get_result_of_compare_method(h, p) -> TEN:  # MMSE beamformer
        h = h[0] + 1j * h[1]
        k, n = h.shape
        eye_mat = th.eye(n, dtype=h.dtype, device=h.device)
        w = th.linalg.solve(eye_mat * k / p + th.conj(th.transpose(h, 0, 1)) @ h, th.conj(th.transpose(h, 0, 1)))
        w = w / (th.norm(w, dim=0, keepdim=True) * k ** 0.5)  # w.shape == [K, N]
        return th.stack((w.real, w.imag), dim=0)  # return.shape == [2, K, N]


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''init'''
    dim = 2, 8, 8

    '''train'''
    train_times = 1000
    lr = 1e-3
    unroll = 16  # step of Hamilton Term
    num_opt = 64
    hid_dim = 40

    '''eval'''
    eval_gap = 128
    num_evals = 10

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = ObjectiveMISO(dim=dim, device=device)
    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerOpti(hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    time_start = time.time()

    '''loop'''
    for i in range(train_times):
        opt_train(obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 1:
            loss_of_p_evals = []
            for p_eval in obj_task.p_evals:
                losses = []
                for h_eval in obj_task.h_evals[:num_evals]:
                    h_scale = h_eval * (p_eval ** 0.5)
                    obj_task.args = (h_scale,)
                    best_result, min_loss = opt_eval(obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                                                     num_opt=num_opt * 2, device=device)
                    losses.append(-min_loss.item())
                loss_of_p_evals.append(sum(losses) / len(losses))
            time_used = round((time.time() - time_start))
            print(f"{'L2O':>8} {loss_of_p_evals[0]:>9.3f} {loss_of_p_evals[1]:>9.3f} {loss_of_p_evals[2]:>9.3f}    "
                  f"TimeUsed {time_used:9}")

    loss_of_p_evals = []
    for p_eval in obj_task.p_evals:
        losses = []
        for h_eval in obj_task.h_evals:
            h_scale = h_eval * (p_eval ** 0.5)
            obj_task.args = (h_scale,)
            best_result, _min_loss = opt_eval(obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                                              num_opt=num_opt * 2, device=device)
            min_loss = obj_task.get_objective(best_result, h_scale)  # todo re-calculate loss
            losses.append(-min_loss.item())
        loss_of_p_evals.append(sum(losses) / len(losses))
    time_used = round((time.time() - time_start))
    print(f"{'L2O':>8} {loss_of_p_evals[0]:>9.3f} {loss_of_p_evals[1]:>9.3f} {loss_of_p_evals[2]:>9.3f}    "
          f"TimeUsed {time_used:9}")


if __name__ == '__main__':
    train_optimizer()