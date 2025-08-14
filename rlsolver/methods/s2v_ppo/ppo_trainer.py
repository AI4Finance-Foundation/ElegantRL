# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_log_softmax, scatter_add


class PPOTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.1
        )
        self.scaler = GradScaler(enabled=config.use_amp)
        
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_ratio = config.clip_ratio
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.update_epochs = config.update_epochs
        self.max_grad_norm = config.max_grad_norm
        
        # Assume config has debug flag; set to False for production
        self.debug = getattr(config, 'debug', False)
        
    def compute_gae(self, rewards, values, dones, final_value=None):
        """
        计算GAE，支持截断情况
        final_value: 如果轨迹被截断（最后一步done=False），使用此值作为引导
        """
        # Clamp inputs early to prevent overflow in GAE
        rewards = torch.clamp(rewards, -100.0, 100.0)
        values = torch.clamp(values, -100.0, 100.0)
        
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]

                if final_value is not None and not dones[t]:
                    nextvalues = final_value
                else:
                    nextvalues = 0
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            delta = torch.clamp(delta, -10.0, 10.0)  # Clamp delta to avoid explosion
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, trajectories):

        # 如果没有轨迹，返回0损失
        if not trajectories:
            return 0.0
            
        # 预计算总步数
        total_steps = sum(len([s for s in traj['steps'] if 'action' in s]) for traj in trajectories)
        
        # 如果总步数为0，返回0损失
        if total_steps == 0:
            return 0.0
        
        # 预分配张量
        device = self.config.device
        all_actions = torch.zeros(total_steps, dtype=torch.long, device=device)
        all_log_probs = torch.zeros(total_steps, device=device)
        all_rewards = torch.zeros(total_steps, device=device)
        all_values = torch.zeros(total_steps, device=device)
        all_dones = torch.zeros(total_steps, device=device)
        
        # 收集数据和索引映射
        all_data = []
        all_masks = []
        step_idx = 0
        
        # 处理每个轨迹
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            # 分离步骤和可能的最终价值
            steps = [s for s in traj['steps'] if 'action' in s]
            final_value_item = next((s for s in traj['steps'] if 'final_value' in s), None)
            final_value = final_value_item['final_value'] if final_value_item else None
            
            if not steps:
                continue
                
            # 收集轨迹数据
            traj_start = step_idx
            for step in steps:
                # 创建Data对象时数据已在GPU上
                data = Data(x=step['x'], edge_index=step['edge_index'])
                all_data.append(data)
                all_masks.append(step['valid_actions_mask'])
                
                # 填充预分配的张量
                all_actions[step_idx] = step['action']
                all_log_probs[step_idx] = step['log_prob']
                all_rewards[step_idx] = step['reward']
                all_values[step_idx] = step['value']
                all_dones[step_idx] = step['done']
                step_idx += 1
            traj_end = step_idx
            
            # 为这个轨迹计算GAE
            traj_rewards = all_rewards[traj_start:traj_end]
            traj_values = all_values[traj_start:traj_end]
            traj_dones = all_dones[traj_start:traj_end]
            

            advantages, returns = self.compute_gae(
                traj_rewards, traj_values, traj_dones, final_value
            )
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # 合并所有轨迹的优势和回报
        advantages = torch.cat(all_advantages)
        returns = torch.cat(all_returns)
        

        has_invalid = False
        if torch.isnan(all_rewards[:step_idx]).any() or torch.isinf(all_rewards[:step_idx]).any():
            has_invalid = True
        if torch.isnan(all_values[:step_idx]).any() or torch.isinf(all_values[:step_idx]).any():
            has_invalid = True
        if torch.isnan(all_log_probs[:step_idx]).any() or torch.isinf(all_log_probs[:step_idx]).any():
            has_invalid = True
        

        if dist.is_initialized():
            invalid_tensor = torch.tensor([1.0 if has_invalid else 0.0], device=device)
            dist.all_reduce(invalid_tensor, op=dist.ReduceOp.MAX)
            has_invalid = invalid_tensor.item() > 0
        
        if has_invalid:
            return 0.0
        
        # 标准化优势
        with torch.no_grad():
            adv_mean = torch.nan_to_num(advantages.mean(), nan=0.0)
            adv_std = torch.nan_to_num(advantages.std(), nan=1.0)
            advantages = (advantages - adv_mean) / (adv_std + 1e-5)
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # 只使用实际的步数
        total_steps = step_idx
        all_actions = all_actions[:total_steps]
        all_log_probs = all_log_probs[:total_steps]
        

        if dist.is_initialized():

            indices = torch.randperm(total_steps, device=device)
            dist.broadcast(indices, src=0)
        else:
            indices = torch.randperm(total_steps, device=device)
        
        total_loss = 0
        num_updates = 0
        
        for epoch in range(self.update_epochs):
            # Shuffle indices - 使用相同的方式确保一致性
            if dist.is_initialized():
                # 从rank 0广播新的shuffle顺序
                indices = indices[torch.randperm(len(indices))]
                dist.broadcast(indices, src=0)
            else:
                indices = indices[torch.randperm(len(indices))]
            
            # 按minibatch处理
            for start_idx in range(0, total_steps, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, total_steps)
                batch_indices = indices[start_idx:end_idx]
                
                # 获取当前批次数据
                batch_data_list = [all_data[idx] for idx in batch_indices.cpu()]
                batch_actions = all_actions[batch_indices]
                batch_old_log_probs = all_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 创建批次图
                batch_data = Batch.from_data_list(batch_data_list)
                
                # 收集masks
                batch_masks = [all_masks[idx] for idx in batch_indices.cpu()]
                batch_mask = torch.cat(batch_masks)
                
                # 检查是否有图没有有效动作
                mask_sums = scatter_add(batch_mask.float(), batch_data.batch, dim=0)
                skip_batch = (mask_sums == 0).any().item()
                

                if dist.is_initialized():
                    skip_tensor = torch.tensor([1.0 if skip_batch else 0.0], device=device)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                    skip_batch = skip_tensor.item() > 0
                
                if skip_batch:
                    continue
                
                # 初始化skip标志
                should_skip = False
                
                with autocast(enabled=self.config.use_amp):
                    # 前向传播
                    logits, values = self.model(
                        batch_data.x,
                        batch_data.edge_index,
                        batch_mask,
                        batch_data.batch
                    )
                    
                    # Clamp logits
                    logits = torch.clamp(logits, -1e4, 1e4)
                    
                    # Check logits NaN/Inf
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        should_skip = True
                    
                    if not should_skip:
                        # scatter_log_softmax
                        log_probs_all_nodes = scatter_log_softmax(logits, batch_data.batch, dim=0)
                        
                        # Global actions - 修复：确保ptr在正确的设备上
                        ptr = batch_data.ptr[:-1].to(batch_actions.device)
                        global_actions = ptr + batch_actions
                        new_log_probs = log_probs_all_nodes[global_actions]
                        
                        # PPO loss
                        log_ratio = new_log_probs - batch_old_log_probs
                        log_ratio = torch.clamp(log_ratio, -20.0, 2.0)
                        ratio = torch.exp(log_ratio)
                        
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        clipped_returns = torch.clamp(batch_returns, -100.0, 100.0)
                        clipped_values = torch.clamp(values, -100.0, 100.0)
                        value_loss = F.huber_loss(clipped_values, clipped_returns, delta=1.0)
                        
                        # Entropy
                        probs = torch.clamp(torch.exp(log_probs_all_nodes), 1e-10, 1.0)
                        stable_log_probs = torch.log(probs)
                        entropy = -torch.sum(probs * stable_log_probs * batch_mask) / (batch_data.num_graphs + 1e-8)
                        entropy_loss = -self.entropy_coef * entropy
                        
                        # Total loss
                        loss = policy_loss + self.value_coef * value_loss + entropy_loss
                        
                        # NaN check on loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            should_skip = True
                
                # DDP同步：确保所有进程一致决定是否skip
                if dist.is_initialized():
                    skip_tensor = torch.tensor([1.0 if should_skip else 0.0], device=device)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                    should_skip = skip_tensor.item() > 0
                
                if should_skip:
                    continue
                
                # Optimization
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                num_updates += 1
        
        self.scheduler.step()
        return total_loss / num_updates if num_updates > 0 else 0