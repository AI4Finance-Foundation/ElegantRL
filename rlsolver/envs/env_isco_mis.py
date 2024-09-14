import torch
from rlsolver.methods.iSCO.config.mis_config import *
from torch.func import vmap


class iSCO:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = torch.device(f'cuda:{GPU_ID}'if torch.cuda.is_available() and GPU_ID >= 0 else 'cpu')
        self.chain_length = CHAIN_LENGTH
        self.lam = LAMADA
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def model(self, x):
        gather2src = torch.gather(x, -1, self.edge_from)
        gather2dst = torch.gather(x, -1, self.edge_to)
        penalty = self.lam * torch.sum(gather2src*gather2dst,dim=-1)
        energy = torch.sum(x,dim=-1) - penalty
        grad = torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=torch.ones_like(energy), create_graph=False,
                                   retain_graph=False)[0]
        with torch.no_grad():
            grad = grad.detach()
            energy = energy.detach()
        return energy, grad

    def random_gen_init_sample(self):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))
        return sample

class iSCO_local_search(iSCO):
    def __init__(self, data):
        super().__init__(data)

    def model(self, x):
        gather2src = torch.gather(x, -1, self.edge_from)
        gather2dst = torch.gather(x, -1, self.edge_to)
        penalty = self.lam * torch.sum(gather2src*gather2dst)
        energy = torch.sum(x) - penalty

        return energy

    def flip(self, x, i_n):
        return x * (1 - 2 * i_n) + i_n

    def x2y(self, x, idx_list, traj, I_N, path_length, temperature):
        cur_x = x.clone()
        energy_x = self.model(x)
        energy_cur_x = energy_x

        for step in range(path_length):
            neighbor_x = vmap(self.flip, in_dims=(None, 0))(cur_x, I_N)
            neighbor_x_energy = vmap(self.model, in_dims=0)(neighbor_x).squeeze()
            score_change_x = (neighbor_x_energy - energy_cur_x) / (2 * temperature)
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)
            traj[:, step] = torch.log_softmax(-score_change_x, dim=-1)
            index = torch.multinomial(prob_x_local.exp(), 1).view(-1)
            idx_list[step] = index
            cur_x[index] = 1.0 - cur_x[index]
            energy_cur_x = neighbor_x_energy[index]

        return cur_x, idx_list, traj, energy_x, energy_cur_x

    def y2x(self, x, energy_x, y, energy_y, idx_list, traj, I_N, path_length, temperature):
        with torch.no_grad():
            r_idx = torch.arange(path_length, device=self.device).view(1, -1)
            neighbor_y = vmap(self.flip, in_dims=(None, 0))(y, I_N)
            neighbor_y_energy = vmap(self.model, in_dims=0)(neighbor_y).squeeze()
            score_change_y = -(neighbor_y_energy - energy_y) / (2 * temperature)
            prob_y_local = torch.log_softmax(score_change_y, dim=-1)
            traj[:, path_length] = torch.log_softmax(-score_change_y, dim=-1)

            log_fwd = torch.sum(traj[:, :-1][idx_list, r_idx], dim=-1) + energy_x.view(-1)
            log_backwd = torch.sum(traj[:, 1:][idx_list, r_idx], dim=-1) + energy_y.view(-1)

            log_acc = log_backwd - log_fwd
            accs = torch.clamp(log_acc.exp(), max=1)
            mask = accs >= torch.rand_like(accs)
            new_x = torch.where(mask, y, x)
            energy = torch.where(mask, energy_y, energy_x)

        return new_x, energy, accs

    def step(self, path_length, temperature, sample):
        x = sample.clone()
        idx_list = torch.empty(self.batch_size, path_length, device=self.device, dtype=torch.int)
        traj = torch.empty(self.batch_size, self.max_num_nodes, path_length + 1, device=self.device)
        I_N = torch.eye(self.max_num_nodes, device=self.device)
        y, idx_list, traj, energy_x, energy_y = self.x2y(x, idx_list, traj, I_N, path_length, temperature)
        new_x, energy, accs = self.y2x(x, energy_x, y, energy_y, idx_list, traj, I_N, path_length, temperature)
        accs = torch.mean(accs)
        return new_x, energy, accs