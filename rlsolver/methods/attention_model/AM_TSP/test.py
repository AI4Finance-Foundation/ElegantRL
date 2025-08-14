"""
TSP模型超参数搜索方案
采用两阶段搜索策略：快速筛选 + 精细调优

修复说明：
1. 修复了config字典转换为Namespace对象的问题
2. 基于你的原始参数(embedding=128, lr=3e-4, batch=64)调整搜索范围
3. 减少搜索空间，聚焦在你的工作参数附近
4. 保持训练集大小为10000，与你的原始设置一致
"""

import os
import json
import time
import torch
import torch.multiprocessing as mp
from itertools import product
from datetime import datetime
from argparse import Namespace

# 导入必要的模块
from models import TSPSolver
from dataset import TSPDataset
from train import setup_ddp, cleanup
from trainer import DistributedTSPTrainer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class HyperparameterSearch:
    """两阶段超参数搜索"""
    
    def __init__(self):
        # 第一阶段：基于你的原始参数进行调整
        self.stage1_params = {
            'embedding_size': [64, 128, 256],  # 保留128作为中心
            'hidden_size': [64, 128, 256],     # 保留128作为中心
            'n_head': [2, 4, 8],               # 保留4作为中心
            'lr': [1e-4, 3e-4, 6e-4],          # 围绕3e-4调整
            'batch_size': [32, 64, 128],      # 围绕64调整
            'C': [5.0, 10.0, 15.0]             # 围绕10.0调整
        }
        
        # 第二阶段：精细调优参数（基于第一阶段结果动态生成）
        self.stage2_params = {}
        
        # 固定参数
        self.fixed_params = {
            'seq_len': 30,
            'grad_clip': 1.5,
            'beta': 0.9,
            'seed': 111,
            'num_workers': 4,
            'use_cuda': True
        }
        
        # 第一阶段训练配置（快速评估）
        self.stage1_config = {
            'num_epochs': 20,  # 较少的训练轮数
            'num_tr_dataset': 10000,  # 使用你的原始训练集大小
            'num_te_dataset': 1000,   # 较小的测试集
            'eval_interval': 5  # 评估间隔
        }
        
        # 第二阶段训练配置（完整训练）
        self.stage2_config = {
            'num_epochs': 50,
            'num_tr_dataset': 10000,  # 保持和你原始设置一致
            'num_te_dataset': 2000,   # 保持和你原始设置一致
            'eval_interval': 10
        }
        
        # 结果保存路径
        self.results_dir = f"hyperparam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def generate_stage1_configs(self):
        """生成第一阶段的参数组合"""
        # 使用智能采样减少搜索空间
        configs = []
        
        # 策略1：embedding_size 和 hidden_size 保持相同或相近
        for embed in self.stage1_params['embedding_size']:
            for lr in self.stage1_params['lr']:
                for n_head in self.stage1_params['n_head']:
                    for batch_size in self.stage1_params['batch_size']:
                        for C in self.stage1_params['C']:
                            config = {
                                'embedding_size': embed,
                                'hidden_size': embed,  # 保持相同
                                'n_head': n_head,
                                'lr': lr,
                                'batch_size': batch_size,
                                'C': C,
                                **self.fixed_params,
                                **self.stage1_config
                            }
                            # 检查 n_head 能否整除 hidden_size
                            if embed % n_head == 0:
                                configs.append(config)
        
        # 限制搜索数量，选择代表性配置
        if len(configs) > 20:  # 减少搜索数量，因为已有较好的基准参数
            # 优先包含原始参数配置
            original_config = None
            for cfg in configs:
                if (cfg['embedding_size'] == 128 and 
                    cfg['hidden_size'] == 128 and 
                    cfg['n_head'] == 4 and 
                    abs(cfg['lr'] - 3e-4) < 1e-6 and 
                    cfg['batch_size'] == 64 and 
                    cfg['C'] == 10.0):
                    original_config = cfg
                    break
            
            # 随机采样其他配置
            import random
            random.seed(42)
            other_configs = [c for c in configs if c != original_config]
            sampled = random.sample(other_configs, 19 if original_config else 20)
            
            # 确保包含原始配置
            if original_config:
                configs = [original_config] + sampled
            else:
                configs = sampled
            
        return configs
    
    def generate_stage2_configs(self, top_configs):
        """基于第一阶段结果生成第二阶段的精细搜索配置"""
        configs = []
        
        for base_config in top_configs[:3]:  # 选择前3个最佳配置
            # 在最佳参数附近进行精细搜索
            lr_variations = [base_config['lr'] * 0.7, base_config['lr'], base_config['lr'] * 1.3]
            c_variations = [base_config['C'] * 0.9, base_config['C'], base_config['C'] * 1.1]
            
            for lr in lr_variations:
                for C in c_variations:
                    config = base_config.copy()
                    config.update({
                        'lr': lr,
                        'C': C,
                        **self.stage2_config
                    })
                    configs.append(config)
        
        # 去重
        unique_configs = []
        seen = set()
        for cfg in configs:
            key = (cfg['embedding_size'], cfg['hidden_size'], cfg['n_head'], 
                   cfg['batch_size'], round(cfg['lr'], 10), round(cfg['C'], 10))
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)
                
        return unique_configs
    
    def train_single_config(self, config, stage, config_id):
        """训练单个配置"""
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No GPUs available")
            
        # 启动分布式训练
        result = mp.Manager().dict()
        mp.spawn(
            self._train_worker,
            args=(world_size, config, stage, config_id, result),
            nprocs=world_size,
            join=True
        )
        
        return dict(result)
    
    def _train_worker(self, rank, world_size, config, stage, config_id, result):
        """分布式训练工作进程"""
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        
        # 将配置字典转换为命名空间对象
        args = Namespace(**config)
        
        # 创建数据集
        train_dataset = TSPDataset(args.seq_len, args.num_tr_dataset)
        test_dataset = TSPDataset(args.seq_len, args.num_te_dataset)
        
        # 创建分布式采样器
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=args.num_workers
        )
        
        eval_loader = DataLoader(
            test_dataset,
            batch_size=args.num_te_dataset,
            shuffle=False,
            pin_memory=True
        )
        
        # 创建模型
        model = TSPSolver(
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            seq_len=args.seq_len,
            n_head=args.n_head,
            C=args.C
        ).cuda(rank)
        
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        # 创建训练器 - 现在传入的是Namespace对象而不是字典
        trainer = DistributedTSPTrainer(model, args, rank, world_size)
        
        # 初始化基线
        trainer.initialize_baseline(train_loader)
        
        # 记录训练开始时间
        start_time = time.time()
        best_eval_reward = float('inf')
        
        # 训练循环
        for epoch in range(args.num_epochs):
            # 训练一个epoch
            avg_loss, avg_reward = trainer.train_epoch(train_loader, epoch)
            
            # 定期评估
            if (epoch + 1) % args.eval_interval == 0:
                eval_reward, _ = trainer.evaluate(eval_loader)
                
                if rank == 0 and eval_reward is not None:
                    if eval_reward < best_eval_reward:
                        best_eval_reward = eval_reward
                        
                    print(f"[Stage {stage} - Config {config_id}] Epoch {epoch+1}: "
                          f"Train Loss={avg_loss:.4f}, Train Reward={avg_reward:.4f}, "
                          f"Eval Reward={eval_reward:.4f}")
        
        # 最终评估
        final_eval_reward, _ = trainer.evaluate(eval_loader)
        
        # 记录结果（仅rank 0）
        if rank == 0:
            training_time = time.time() - start_time
            result.update({
                'config': config,
                'final_eval_reward': final_eval_reward,
                'best_eval_reward': best_eval_reward,
                'training_time': training_time,
                'config_id': config_id,
                'stage': stage
            })
        
        cleanup()
    
    def run_search(self):
        """执行完整的超参数搜索"""
        print("=" * 60)
        print("开始超参数搜索")
        print("=" * 60)
        
        # 添加原始配置作为基准
        print("\n提示：搜索将包含你的原始参数配置作为基准对比")
        print("原始配置：embedding=128, n_head=4, lr=3e-4, batch=64, C=10.0")
        
        # 第一阶段：快速筛选
        print("\n第一阶段：快速筛选")
        print("-" * 40)
        
        stage1_configs = self.generate_stage1_configs()
        print(f"生成了 {len(stage1_configs)} 个配置进行第一阶段搜索")
        
        stage1_results = []
        for i, config in enumerate(stage1_configs):
            print(f"\n训练配置 {i+1}/{len(stage1_configs)}")
            print(f"  参数: embed={config['embedding_size']}, n_head={config['n_head']}, "
                  f"lr={config['lr']:.1e}, batch={config['batch_size']}, C={config['C']}")
            
            result = self.train_single_config(config, stage=1, config_id=i)
            stage1_results.append(result)
            
            # 保存中间结果
            self.save_results(stage1_results, 'stage1_results.json')
        
        # 根据评估奖励排序（越小越好）
        stage1_results.sort(key=lambda x: x['best_eval_reward'])
        
        print("\n第一阶段最佳配置:")
        for i, result in enumerate(stage1_results[:5]):
            print(f"{i+1}. Reward={result['best_eval_reward']:.4f}, "
                  f"Time={result['training_time']:.1f}s")
            print(f"   参数: embed={result['config']['embedding_size']}, "
                  f"n_head={result['config']['n_head']}, "
                  f"lr={result['config']['lr']:.1e}, "
                  f"batch={result['config']['batch_size']}, "
                  f"C={result['config']['C']}")
        
        # 第二阶段：精细调优
        print("\n第二阶段：精细调优")
        print("-" * 40)
        
        top_configs = [r['config'] for r in stage1_results[:3]]
        stage2_configs = self.generate_stage2_configs(top_configs)
        print(f"生成了 {len(stage2_configs)} 个配置进行第二阶段搜索")
        
        stage2_results = []
        for i, config in enumerate(stage2_configs):
            print(f"\n训练配置 {i+1}/{len(stage2_configs)}")
            result = self.train_single_config(config, stage=2, config_id=i)
            stage2_results.append(result)
            
            # 保存中间结果
            self.save_results(stage2_results, 'stage2_results.json')
        
        # 最终结果排序
        stage2_results.sort(key=lambda x: x['best_eval_reward'])
        
        # 输出最终推荐配置
        print("\n" + "=" * 60)
        print("搜索完成！最优配置：")
        print("=" * 60)
        
        best_config = stage2_results[0]['config']
        print(f"最佳评估奖励: {stage2_results[0]['best_eval_reward']:.4f}")
        print(f"训练时间: {stage2_results[0]['training_time']:.1f}秒")
        print("\n参数配置:")
        for key, value in best_config.items():
            if key not in ['num_epochs', 'num_tr_dataset', 'num_te_dataset', 'eval_interval']:
                print(f"  {key}: {value}")
        
        # 检查是否为原始配置
        is_original = (best_config['embedding_size'] == 128 and 
                      best_config['n_head'] == 4 and 
                      abs(best_config['lr'] - 3e-4) < 1e-6 and 
                      best_config['batch_size'] == 64 and 
                      best_config['C'] == 10.0)
        
        if is_original:
            print("\n提示：最优配置与你的原始参数相同，说明原始参数已经很好！")
        else:
            print("\n提示：找到了比原始参数更好的配置！")
        
        # 保存最终结果
        final_results = {
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'best_config': best_config,
            'search_time': time.time()
        }
        self.save_results(final_results, 'final_results.json')
        
        # 生成配置文件
        self.generate_config_file(best_config)
        
        return best_config
    
    def save_results(self, results, filename):
        """保存搜索结果"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def generate_config_file(self, config):
        """生成最优配置文件"""
        config_content = f"""# 最优参数配置
# 通过超参数搜索自动生成

embedding_size = {config['embedding_size']}
hidden_size = {config['hidden_size']}
n_head = {config['n_head']}
C = {config['C']}
seq_len = {config['seq_len']}
num_tr_dataset = 10000  # 训练集大小
num_te_dataset = 2000   # 测试集大小

num_epochs = 100  # 完整训练轮数
batch_size = {config['batch_size']}
lr = {config['lr']}
grad_clip = {config['grad_clip']}
beta = {config['beta']}

use_cuda = True
num_workers = {config['num_workers']}
seed = {config['seed']}
save_freq = 10
"""
        
        filepath = os.path.join(self.results_dir, 'optimal_config.py')
        with open(filepath, 'w') as f:
            f.write(config_content)
        
        print(f"\n最优配置已保存到: {filepath}")


if __name__ == '__main__':
    # 执行超参数搜索
    searcher = HyperparameterSearch()
    best_config = searcher.run_search()
    
    print("\n搜索完成！")
    print(f"结果保存在: {searcher.results_dir}/")
    print("你可以使用生成的 optimal_config.py 替换原始的 config.py 进行完整训练。")