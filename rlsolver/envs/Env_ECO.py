import torch
from torch.func import vmap

from rlsolver.methods.config_env import *
from rlsolver.methods.util import *


class ECO_maxcut:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_graph_partitioning:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_number_partitioning:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_TSP:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_graph_coloring:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_portfolio_allocation:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_TNCO:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_BILP:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_VRP:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_MIS:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_set_cover:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y


class ECO_knapsack:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.max_num_nodes = params_dict['num_nodes']
        self.num_edges = params_dict['num_edges']
        self.edge_from = params_dict['edge_from']
        self.edge_to = params_dict['edge_to']

    def random_gen_init_sample(self, params_dict):
        sample = torch.bernoulli(torch.full((BATCH_SIZE, self.max_num_nodes,), 0.5, device=self.device))

        return sample

    def step(self, x, path_length, temperature):
        ll_x, y, trajectory = self.proposal(x, path_length, temperature)
        ll_x2y = trajectory['ll_x2y']
        ll_y, ll_y2x = self.ll_y2x(
            trajectory, y, temperature)
        log_acc = torch.clamp(ll_y + ll_y2x - ll_x - ll_x2y, max=0.0)
        y = self.select_sample(log_acc, x, y)

        return y, ll_y * temperature, log_acc.exp()

    def proposal(self, x, path_length, temperature):
        ll_x, log_prob = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = multinomial(log_prob, path_length)
        mask = selected_idx['selected_mask']

        new_val = 1 - x
        y = x * (1 - mask) + mask * new_val
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return ll_x, y, trajectory

    def get_local_dist(self, sample, temperature):
        x = sample.clone().detach().requires_grad_(True)
        energy_x = vmap(self.model, in_dims=(0, None))(x, temperature)
        grad_x = torch.autograd.grad(energy_x, x, grad_outputs=torch.ones_like(energy_x), retain_graph=False,
                                     create_graph=False)[0]
        grad_x = grad_x.detach()
        energy_x = energy_x.detach()
        with torch.no_grad():
            delta_x = 1 - x * 2
            score_change_x = (delta_x * grad_x) / 2
            prob_x_local = torch.log_softmax(score_change_x, dim=-1)

        return energy_x, prob_x_local

    def ll_y2x(self, forward_trajectory, y, temperature):
        ll_y, log_prob = self.get_local_dist(
            y, temperature)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y, ll_y2x

    def model(self, sample, temperature):
        delta_x = (sample * 2 - 1)
        gather2src = torch.gather(delta_x, -1, self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut, dim=-1) / temperature

        return energy

    def select_sample(self, log_acc, x, y):
        y, acc = mh_step(log_acc, x, y)

        return y
