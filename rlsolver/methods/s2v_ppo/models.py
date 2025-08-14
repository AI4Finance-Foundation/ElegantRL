# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import aggr


class S2VDQN_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(S2VDQN_layer, self).__init__()
        self.theta_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.theta_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta_3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta_4 = nn.Linear(1, hidden_dim, bias=False)
        self.aggr = aggr.SumAggregation()
    
    def forward(self, x, edge_index, edge_attr, node_embedding):
        row, col = edge_index
        
        node_feature_embedding = self.theta_1(x)
        
        node_embedding_aggr = self.aggr(
            node_embedding[col],
            row,
            dim=0,
            dim_size=len(x)
        )
        
        edge_feature_embedding = F.relu(self.theta_4(edge_attr))
        
        edge_embedding_aggr = self.aggr(
            edge_feature_embedding,
            row,
            dim=0,
            dim_size=len(x)
        )
        
        node_embedding = F.relu(
            node_feature_embedding + 
            self.theta_2(node_embedding_aggr) + 
            self.theta_3(edge_embedding_aggr)
        )
        
        return node_embedding


class S2VEncoder(nn.Module):
    """S2V编码器，替代LinearEncoder"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.hop = getattr(config, 'hop', 3)  # 默认3跳
        
        # S2V层
        self.layers = nn.ModuleList([
            S2VDQN_layer(config.node_feature_dim, config.hidden_dim) 
            for _ in range(self.hop)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, edge_index, batch=None):
        # 边权重都是1维的
        edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        
        # 初始化节点嵌入
        node_embedding = torch.zeros(len(x), self.hidden_dim, device=x.device)
        
        # 通过S2V层
        for layer in self.layers:
            node_embedding = layer(x, edge_index, edge_attr, node_embedding)
            node_embedding = self.dropout(node_embedding)
        
        return node_embedding


class Actor(nn.Module):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, h, valid_actions_mask=None):
        logits = self.mlp(h).squeeze(-1)
        if valid_actions_mask is not None:
            logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        return logits


class Critic(nn.Module):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, h, batch_idx=None):
        if batch_idx is None:
            mean_h = h.mean(dim=0, keepdim=True)
            max_h = h.max(dim=0, keepdim=True)[0]
            sum_h = h.sum(dim=0, keepdim=True)
        else:
            mean_h = global_mean_pool(h, batch_idx)
            max_h = global_max_pool(h, batch_idx)
            sum_h = global_add_pool(h, batch_idx)
            
        graph_emb = torch.cat([mean_h, max_h, sum_h], dim=-1)
        return self.mlp(graph_emb).squeeze(-1)


class PPOLinearModel(nn.Module):
    """使用S2V编码器的PPO模型"""
    def __init__(self, config):
        super().__init__()
        self.encoder = S2VEncoder(config)  
        self.actor = Actor(config.hidden_dim)
        self.critic = Critic(config.hidden_dim)
        
    def forward(self, x, edge_index, valid_actions_mask=None, batch_idx=None):
        h = self.encoder(x, edge_index, batch_idx)
        logits = self.actor(h, valid_actions_mask)
        value = self.critic(h, batch_idx)
        return logits, value