# evaluate.py
# 此文件应放置在 rlsolver/methods/s2v_ppo/ 目录下
import os
import torch
import networkx as nx
from pathlib import Path
from tqdm import tqdm

from models import PPOLinearModel
from config import Config
from env import MaxCutEnv
from torch_geometric.data import Data, Batch


def load_graph_from_file(filepath):
    """从文件加载图数据"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    n, m = map(int, lines[0].strip().split())
    
    # 创建图
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    
    # 添加边
    for i in range(1, m + 1):
        parts = lines[i].strip().split()
        u, v = int(parts[0]) - 1, int(parts[1]) - 1  # 转换为0索引
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        # 确保节点索引在有效范围内
        if 0 <= u < n and 0 <= v < n:
            graph.add_edge(u, v, weight=weight)
    
    # 转换为张量格式
    edges = []
    weights = []
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        edges.extend([[u, v], [v, u]])
        weights.extend([weight, weight])
    
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32) if weights else torch.empty(0, dtype=torch.float32)
    
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': n
    }


def evaluate_single_graph(model, graph_data, config):
    """对单个图进行评估"""
    # 创建环境
    env = MaxCutEnv(graph_data, config)
    state = env.reset()
    
    # 贪婪策略运行直到结束
    done = False
    while not done:
        with torch.no_grad():
            # 创建Data对象并转换为Batch
            data = Data(x=state['x'], edge_index=state['edge_index'])
            batch = Batch.from_data_list([data]).to(config.device)
            valid_mask = state['valid_actions_mask']
            
            # 模型推理
            logits, _ = model(
                batch.x, 
                batch.edge_index, 
                valid_mask, 
                batch.batch
            )
            
            # 选择最佳动作（贪婪）
            masked_logits = logits.clone()
            masked_logits[~valid_mask] = float('-inf')
            action = masked_logits.argmax().item()
            
            # 执行动作
            state, _, done, info = env.step(action)
    

    best_z = env.best_z.cpu().numpy()
    assignment = (best_z == 1).astype(int) + 1
    
    return assignment, info['best_cut']


def main():
    # 配置
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    config.tabu_tenure = 10  
    config.episode_length_multiplier = 2  
    
    # 加载模型
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=config.device)
    
    # 创建模型并加载权重
    model = PPOLinearModel(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功，设备：{config.device}")
    
    # 如果需要debug CUDA错误，可以设置环境变量
    if os.environ.get('CUDA_LAUNCH_BLOCKING'):
        print("CUDA_LAUNCH_BLOCKING已启用，便于调试")
    
    # 数据目录（相对于 rlsolver/methods/maxcut）
    data_root = '../../data'
    result_root = '../../result'
    
    # 获取所有测试文件
    test_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.txt') and 'dataset_info' not in file:
                rel_path = os.path.relpath(root, data_root)
                test_files.append((os.path.join(root, file), rel_path, file))
    
    print(f"找到 {len(test_files)} 个测试文件")
    
    # 处理每个文件
    for filepath, rel_dir, filename in tqdm(test_files, desc="评估进度"):
        try:
            # 清理CUDA缓存
            if config.device == 'cuda':
                torch.cuda.empty_cache()
            
            # 加载图
            graph_data = load_graph_from_file(filepath)
            
            # 评估
            assignment, cut_value = evaluate_single_graph(model, graph_data, config)
            
            # 创建输出目录
            output_dir = os.path.join(result_root, rel_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w') as f:
                for node_id, group in enumerate(assignment, 1):
                    f.write(f"{node_id} {group}\n")
            
            print(f"\n{filename}: 最大割值 = {cut_value}")
            
        except Exception as e:
            continue
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()