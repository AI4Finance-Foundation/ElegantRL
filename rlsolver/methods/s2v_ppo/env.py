# env.py
import torch
import numpy as np
from collections import deque
from torch_scatter import scatter_add


class MaxCutEnv:
    def __init__(self, graph_data, config):
        self.config = config
        self.device = config.device
        self.edge_index = graph_data['edge_index'].to(self.device)
        self.edge_weight = graph_data['edge_weight'].to(self.device)
        self.num_nodes = graph_data['num_nodes']
        self.max_steps = config.episode_length_multiplier * self.num_nodes
        self.tabu_tenure = config.tabu_tenure
        
        # Precompute constants
        self.total_weight = self.edge_weight.sum() / 2
        self.max_possible_cut = self.total_weight
        
        # Compute degrees using scatter_add
        self.degrees = scatter_add(
            self.edge_weight, 
            self.edge_index[0], 
            dim=0, 
            dim_size=self.num_nodes
        )
        

        self.z = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        self.delta_cache = torch.zeros(self.num_nodes, device=self.device)
        self.features_tensor = torch.zeros((self.num_nodes, self.config.node_feature_dim), device=self.device)
        self.valid_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        
        # 状态历史追踪（用于检测重复状态）
        self.state_history = deque(maxlen=50)  
        self.visited_basins = set()  
        
        节点翻转时间追踪
        self.time_since_flip = torch.zeros(self.num_nodes, device=self.device)
        
        self.current_cut = 0
        self.reset()
        
    def reset(self):
        # 重用预分配的张量，使用in-place操作
        if np.random.random() < 0.5:
            self.z.random_(0, 2).mul_(2).sub_(1)
        else:
            self.z.fill_(1)
            for _ in range(self.num_nodes // 2):
                gains = self._compute_all_deltas_gpu()
                best_v = gains.argmax()
                if gains[best_v] <= 0:
                    break
                self.z[best_v] = -self.z[best_v]
        
        self.step_count = 0
        self.tabu_list = deque(maxlen=self.tabu_tenure)
        self.current_cut = self._compute_cut_gpu()
        self._compute_all_deltas_gpu_inplace()  # 使用in-place版本
        self.best_cut = self.current_cut
        self.best_z = self.z.clone()
        self.no_improve_count = 0
        
        # 重置历史追踪
        self.state_history.clear()
        self.visited_basins.clear()
        
        # 重置时间追踪
        self.time_since_flip.zero_()
        
        return self._get_state()
    
    def _get_state_hash(self):
        """计算当前状态的哈希值（用于检测重复状态）"""
        return hash(self.z.cpu().numpy().tobytes())
    
    def _is_basin(self):
        """检查是否处于盆地状态（所有delta <= 0）"""
        return (self.delta_cache <= 0).all().item()
    
    def _compute_neighbor_contrib(self):
        """Compute Adj @ z using scatter_add"""
        return scatter_add(
            self.edge_weight * self.z.float()[self.edge_index[1]], 
            self.edge_index[0], 
            dim=0, 
            dim_size=self.num_nodes
        )
    
    def _compute_cut_gpu(self):
        neighbor_contrib = self._compute_neighbor_contrib()
        same_side = (self.z.float() * neighbor_contrib).sum()
        return (self.total_weight / 2 - same_side / 4).item()
    
    def _compute_all_deltas_gpu(self):
        neighbor_contrib = self._compute_neighbor_contrib()
        return self.z.float() * neighbor_contrib 
    
    def _compute_all_deltas_gpu_inplace(self):

        neighbor_contrib = self._compute_neighbor_contrib()
        self.delta_cache.copy_(self.z.float() * neighbor_contrib)
    
    def get_valid_actions_mask(self):

        self.valid_mask.fill_(True)
        if self.tabu_tenure > 0 and self.tabu_list:
            tabu_indices = torch.tensor(list(self.tabu_list), device=self.device, dtype=torch.long)
            invalid = self.delta_cache[tabu_indices] <= 0
            self.valid_mask[tabu_indices[invalid]] = False
        return self.valid_mask
    
    def _get_node_features(self):

        self.features_tensor[:, 0] = self.z.float()  # 节点分配
        self.features_tensor[:, 1] = self.delta_cache / (self.max_possible_cut + 1e-8)  # delta增益归一化
        
        if self.tabu_tenure > 0:
            self.features_tensor[:, 2].zero_()  # tabu标记
            if self.tabu_list:
                tabu_indices = torch.tensor(list(self.tabu_list), device=self.device, dtype=torch.long)
                self.features_tensor[tabu_indices, 2] = 1.0
        

        self.features_tensor[:, 3] = self.time_since_flip / self.max_steps  # 自上次翻转的时间
        self.features_tensor[:, 4] = (self.delta_cache > 0).sum() / self.num_nodes  # 正delta比例
        
        return self.features_tensor
    
    def _get_state(self):
        return {
            'x': self._get_node_features(),
            'edge_index': self.edge_index,
            'valid_actions_mask': self.get_valid_actions_mask()
        }
    
    def step(self, action):
        delta = self.delta_cache[action].item()
        old_cut = self.current_cut
        self.z[action] = -self.z[action]
        self.current_cut += delta 
        
        # Vectorized delta update
        self._update_deltas_gpu(action)
        
        new_cut = self.current_cut
        self.step_count += 1
        if self.tabu_tenure > 0:
            self.tabu_list.append(action)
        
        # 更新时间追踪
        self.time_since_flip += 1.0
        self.time_since_flip[action] = 0  # 重置翻转节点的时间
        
        # 检查是否访问新状态
        state_hash = self._get_state_hash()
        visiting_new_state = state_hash not in self.state_history
        
        # BLS变体奖励机制
        reward = 0.0
        
        # 1. BLS奖励
        if new_cut > self.best_cut:
            reward = (new_cut - self.best_cut) / (new_cut - self.best_cut + 0.1)
            self.best_cut = new_cut
            self.best_z = self.z.clone()
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        
        # 2. 滞留惩罚
        if not visiting_new_state:
            reward -= self.config.stag_punishment
        
        # 3. 盆地奖励
        if self._is_basin():
            basin_hash = self._get_state_hash()
            if basin_hash not in self.visited_basins:
                reward += 0.01
                self.visited_basins.add(basin_hash)
        
        # 更新状态历史
        self.state_history.append(state_hash)
        
        done = self.step_count >= self.max_steps
        
        info = {
            'cut_value': new_cut,
            'best_cut': self.best_cut
        }
        
        return self._get_state(), reward, done, info
    
    def _update_deltas_gpu(self, action):
        """Vectorized delta update without loops"""
        mask = (self.edge_index[0] == action)
        neighbors = self.edge_index[1][mask]
        weights = self.edge_weight[mask]
        
        # Vectorized update
        update_values = 2 * self.z[neighbors].float() * self.z[action].float() * weights
        self.delta_cache.index_add_(0, neighbors, update_values)
        self.delta_cache[action] = -self.delta_cache[action]