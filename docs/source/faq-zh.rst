FAQ


^^^^^^^^
问题 1：在强化学习代码中，对log值裁剪到 -20 到 +2 之间是在进行什么操作？为什么要裁剪到这两个值之间？
^^^^^^^^

在强化学习中，我们举两类对log值进行裁剪的例子：

- 对随机策略的动作的高斯分布的方差的log值 `action_std_log` 进行裁剪

- 对正态分布中对应的概率的log值 (log probability) `logprob` 进行裁剪

简单说，就是相对于正态分布 N~(0, 1) 来说，一个高斯分布的方差的log值如果超过 (-20, +2) 这个区间，那么：

- 如果log值小于 -20，那么这个高斯分布的方差特别小，相当于没有方差，接近于一个确定的数值。

- 如果log值大于 +2，那么这个高斯分布的方差特别大，相当于在接近均值附近是均匀分布。

有空我就展开讲一讲。

-----------------
对随机策略的动作的高斯分布的方差的log值 `action_std_log` 进行裁剪
-----------------
对应代码是  `action_std = self.net_action_std(t_tmp).clip(-20, 2).exp()`, 可以在 `elegantrl/net.py` 里找到。

SAC算法的 `alpha_log` 也能进行类似的裁剪

还可以讲一讲 强化学习里，把权重处理成 log 形式再进行梯度优化。

有空我就展开讲一讲。或者你们来补充（2022-06-08 18:01:54）

-----------------
对正态分布中对应的概率的log值 (log probability) `logprob` 进行裁剪
-----------------
对应代码是  `logprob = logprob.clip(-20, 2)`, 有可能在 `elegantrl/agent/` 里的随机策略梯度算法里找到，因为随机策略梯度算法会用到 `logprob`。

有空我就展开讲一讲。或者你们来补充（2022-06-08 18:01:54）


^^^^^^^^
问题：On-policy 和 off-policy 的区别是什么？
^^^^^^^^
若行为策略和目标策略相同，则是on-policy,若不同则为off-policy

有空我就展开讲一讲。


^^^^^^^^
问题： elegantrl RLlib SB3 对比
^^^^^^^^

RLlib 的优势：
- 1. 他们有ray 可以调度多卡之间的传输，多卡的时候选择 生产者-消费者 模式，保证把计算资源用满
- 2. RLlib的复杂代码把RL过程抽象出来了，他们可以选择 TensorFlow 或者 PyTorch 作为深度学习的后端

RLlib的劣势：
- 1. 让 生产者worker 和 消费者 learner 异步的方案，数据不够新，虽然计算资源用尽了，但是计算效率降低了
- 2. 现在大家都 PyTorch ，RLlib 的代码太复杂了，用起来有门槛，反而不容易用

ELegantRL 在单卡上和 RLlib比较：
- 1. 论文写了，我们让 一张GPU运行完worker，就让 learner直接用 worker收集到的data，数据不用挪动，因此快。
- 2. 我们的代码从 worker 到 learner都支持了 vectorized env，（我不清楚现在RLlib 的worker 是否支持 vectorized env ，但他们的 learner 支持不了）
- 3. 我们还开发了 vwap 的 vectorized env，而不只是 stable baselines3 或者 天授的EnvPool 那种 subprocessing vectorized env

ELegantRL 在多卡上和RLlib比较：
只能说是各有优劣，不能说谁的方案更适合某个 DRL算法或者某个 任务。
我们在金融任务上，使用了 PPO+Podracer，而不是 RLlib 的 Producter&Comsumer 的模式，让PPO算法的数据利用效率更高，而且我们还套了一层 遗传算法在外面方便跳出局部最优，达到更好的次优。


比SB3更稳定，是因为，ELegantRL 和 sb3在以下两点差别明显：
- 1. 我们还开发了 vmap 的 vectorized env，而不只是 stable baselines3 或者 天授的EnvPool 那种 subprocessing vectorized env，用GPU做仿真环境的并行（StockTradingVecEnv），采集数据量多了2个数量级以上，数据多，所以训练稳定
- 2. 我们用了 H term，这个真的有用，可以让训练变稳定：在根据符合贝尔曼公式Q值的优化方向的基础上，再使用一个 H term 找出另一个优化方向，两个优化方向同时使用，更不容易掉进局部最优，所以稳定（可惜当前ELegantRL库 只有 半年前的代码支持了 H term，还需要人手把 Hterm 的代码升级到 2023年2月份的版本）

比SB3快：
- 1. 我们的 ReplayBuffer优化过，按顺序储存 state，所以不需要重复保存 state_t 和 state_t+1，再加上我们的ReplayBuffer都是 PyTorch tensor 格式 + 指针，抽取数据没有用PyTorch自带的 dataLoader，而是自己写的，因此快
- 2. 我们的 worker 和 learner 都有 针对 vectorized env 的优化，sb3没有
- 3. 我们给FinRL任务以及 RLSolver任务 开发了 GPU并行仿真环境，sb3 没有
	
  
  
  
