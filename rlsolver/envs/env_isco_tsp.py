import torch

from rlsolver.methods.iSCO.config.config_tsp import *
from rlsolver.methods.iSCO.util import math_util


class iSCO:
    def __init__(self, params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE, device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE, device=self.device)
        self.num_nodes = params_dict['num_nodes']
        self.distance = params_dict['distance']
        self.nearest_indices = params_dict['nearest_indices']
        self.random_indices = params_dict['random_indices']

    def step(self, x, path_length, temperature):
        cur_x = x.clone()
        traj = torch.zeros((BATCH_SIZE, 3, path_length), dtype=torch.float, device=self.device)
        for i in range(path_length):
            cur_x, logits, trajectory, delta_yx = self.proposal(cur_x, temperature)
            ll_x2y = trajectory['ll_x2y']
            ll_y2x = self.y2x(logits, trajectory)
            traj[:, 0, i], traj[:, 1, i], traj[:, 2, i] = delta_yx, -ll_x2y, ll_y2x
        log_acc = torch.clamp(torch.sum(traj, dim=(1, 2)), max=0.0)
        y, accetped = self.select_sample(log_acc, x, cur_x)

        return y, torch.mean(log_acc.exp())

    def proposal(self, sample, temperature):
        x = sample.clone()
        logits, log_prob, indices, ban_mask, delta_yx = self.get_local_dist(x, temperature)
        selected_idx, ll_selected = math_util.multinomial(log_prob, torch.ones(BATCH_SIZE, dtype=torch.int64, device=self.device))
        logits = logits * (1 - 2 * selected_idx['selected_mask'])
        swap_env_mask, swap_sample_mask = torch.where(((selected_idx['selected_mask'] == 1) & (~ban_mask)) == 1)
        x = self.switch(sample, swap_env_mask, swap_sample_mask, indices)
        trajectory = {
            'll_x2y': torch.sum(ll_selected, dim=-1),
            'selected_idx': selected_idx,
        }

        return x, logits, trajectory, torch.sum(delta_yx * selected_idx['selected_mask'], dim=-1)

    def get_local_dist(self, sample, temperature):
        # log_prob是每点采样的权重，logratio是delta_yx
        x = sample.detach()
        logratio, indices, ban_mask = self.opt_2(x, temperature)
        logratio[ban_mask] = -1e6
        logits = self.apply_weight_function_logscale(logratio)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)

        return logits, log_prob, indices, ban_mask, logratio

    def y2x(self, logits, forward_trajectory):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_mask = forward_trajectory['selected_idx']['selected_mask']
        order_info = forward_trajectory['selected_idx']['perturbed_ll']
        backwd_idx = torch.argsort(order_info, dim=-1)
        log_prob = torch.where(selected_mask.bool(), log_prob, torch.tensor(-1e18))
        backwd_ll = torch.gather(log_prob, dim=-1, index=backwd_idx)
        backwd_mask = torch.gather(selected_mask, dim=-1, index=backwd_idx)
        ll_backwd = math_util.noreplacement_sampling_renormalize(backwd_ll)
        ll_y2x = torch.sum(torch.where(backwd_mask.bool(), ll_backwd, torch.tensor(0.0)), dim=-1)

        return ll_y2x

    def opt_2(self, sample, temperature):
        batch_size = sample.shape[0]
        num_nodes = self.num_nodes  # 节点数量

        # 生成索引掩码，形状为 (batch_size, num_nodes)
        mask = torch.arange(num_nodes, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # 生成随机数，用于选择 nearest_indices 或 random_indices，形状为 (batch_size, num_nodes)
        rand_numbers = torch.rand(batch_size, num_nodes, device=self.device)
        condition = rand_numbers < (K / (K + 1))

        # 从 sample 中索引 nearest_indices 和 random_indices，形状分别为
        # nearest_indices_sample: (batch_size, num_nodes, K)
        # random_indices_sample: (batch_size, num_nodes, num_nodes - K - 1)
        # 得到sample的nearest_indices和random_indices
        nearest_indices_sample = self.nearest_indices[sample]
        random_indices_sample = self.random_indices[
            sample]

        # 生成随机索引，用于从 nearest_indices_sample 和 random_indices_sample 中选择，形状为 (batch_size, num_nodes)
        nearest_rand_indices = torch.randint(0, K, (batch_size, num_nodes), device=self.device)
        random_rand_indices = torch.randint(0, num_nodes - K - 1, (batch_size, num_nodes), device=self.device)

        # 根据随机索引从 nearest_indices_sample 和 random_indices_sample 中选择元素
        nearest_selected_mask = torch.gather(nearest_indices_sample, 2, nearest_rand_indices.unsqueeze(2)).squeeze(
            2)
        random_selected_mask = torch.gather(random_indices_sample, 2, random_rand_indices.unsqueeze(2)).squeeze(2)

        # 根据条件选择 selected_mask，形状为 (batch_size, num_nodes)
        selected_mask = torch.where(condition, nearest_selected_mask, random_selected_mask)

        # 对 sample 进行排序，得到排序后的样本和对应的原始索引，形状均为 (batch_size, num_nodes)
        sorted_sample, sorted_indices = torch.sort(sample, dim=1)

        # 在排序后的样本中查找 selected_mask 的插入位置，形状为 (batch_size, num_nodes)
        sorted_t2_indices = torch.searchsorted(sorted_sample, selected_mask, right=False)

        # 根据插入位置从 sorted_indices 中获取对应的索引，形状为 (batch_size, num_nodes)
        indices = torch.gather(sorted_indices, 1, sorted_t2_indices)

        # 计算 mask 的前后位置，使用取模操作以处理循环，形状均为 (batch_size, num_nodes)
        mask0 = (mask - 1) % num_nodes
        mask1 = (mask + 1) % num_nodes
        mask2 = (mask + 2) % num_nodes

        indices0 = (indices - 1) % num_nodes
        indices1 = (indices + 1) % num_nodes

        # 获取相应位置的样本值，用于条件判断
        sample_mask1 = torch.gather(sample, 1, mask1)
        sample_mask0 = torch.gather(sample, 1, mask0)

        # 条件1和条件2的判断，形状为 (batch_size, num_nodes)
        condition1 = (sample_mask1 == selected_mask)
        condition2 = (sample_mask0 == selected_mask)
        combined_condition = condition1 | condition2

        sample_indices0 = torch.gather(sample, 1, indices0)
        sample_indices1 = torch.gather(sample, 1, indices1)
        sample_indices = torch.gather(sample, 1, indices)

        condition3 = (sample_mask1 == sample_indices0)

        # 准备计算 delta_yx 所需的节点索引
        node_mask = torch.gather(sample, 1, mask)
        node_mask1 = sample_mask1
        node_mask2 = torch.gather(sample, 1, mask2)
        node_indices0 = sample_indices0
        node_indices = sample_indices
        node_indices1 = sample_indices1

        # 计算距离，需将节点索引展开为一维，然后再 reshape 回原来的形状
        def get_distance(u, v):
            return self.distance[u.view(-1), v.view(-1)].view(batch_size, num_nodes)

        # 根据条件计算 delta_yx
        delta_yx = torch.where(
            combined_condition,
            torch.zeros((batch_size, num_nodes), dtype=torch.float, device=self.device),
            torch.where(
                condition3,
                # 当 condition3 为 True 时的计算
                -(
                        get_distance(node_mask, node_mask1) + get_distance(node_indices, node_indices1)
                ) + (
                        get_distance(node_mask, node_indices) + get_distance(node_indices0, node_indices1)
                ),
                # 当所有条件均不满足时的计算
                -(
                        get_distance(node_mask, node_mask1) + get_distance(node_mask1, node_mask2) +
                        get_distance(node_indices0, node_indices) + get_distance(node_indices, node_indices1)
                ) + (
                        get_distance(node_mask, node_indices) + get_distance(node_indices, node_mask2) +
                        get_distance(node_indices0, node_mask1) + get_distance(node_mask1, node_indices1)
                )
            )
        )

        return -delta_yx / temperature, indices, combined_condition

    def switch(self, sample, swap_env_mask, swap_sample_mask, indices):
        x = sample.clone()
        indices = indices[swap_env_mask, swap_sample_mask]
        swap_sample_mask = (swap_sample_mask + 1) % self.num_nodes
        temp = x[swap_env_mask, swap_sample_mask]
        x[swap_env_mask, swap_sample_mask] = x[swap_env_mask, indices]
        x[swap_env_mask, indices] = temp
        return x

    def calculate_distance(self, sample):
        distances = self.distance[sample[:, :-1], sample[:, 1:]]  # (num_envs, num_nodes - 1)
        total_distance = distances.sum(dim=1)  # (num_envs,)
        total_distance += self.distance[sample[:, -1], sample[:, 0]]
        return total_distance

    def random_gen_init_sample(self, params_dict):
        sample = torch.stack([torch.randperm(self.num_nodes, dtype=torch.long, device=self.device) for _ in range(BATCH_SIZE)])
        return sample

    def select_sample(self, log_acc, x, y):
        y, accepted = math_util.mh_step(log_acc, x, y)
        return y, accepted

    def apply_weight_function_logscale(self, logratio):
        logits = logratio / 2
        # logits = th.nn.functional.logsigmoid(logratio)
        return logits
