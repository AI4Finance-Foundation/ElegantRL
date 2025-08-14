# config.py
import torch

class Config:
    # 模型架构
    node_feature_dim = 5  
    hidden_dim = 128
    dropout = 0.1
    hop = 3  # S2V跳数
    
    # PPO超参数
    lr = 2e-4
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.02
    max_grad_norm = 0.5
    
    # 训练参数
    epochs = 500
    batch_size = 8192
    minibatch_size = 256
    update_epochs = 4
    num_parallel_envs = 8
    use_amp = True
    force_reload = False
    
    # 环境参数
    episode_length_multiplier = 2
    tabu_tenure = 10
    no_improve_norm = 100.0
    stag_punishment = 0.01  # 滞留惩罚
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 随机种子
    seed = 39