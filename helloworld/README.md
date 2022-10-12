## ElegantRL helloworld 

Three typical algorithms are presented: 
- Deep Q-Network (DQN): the first DRL algorithm.
- Deep Deterministic Policy Gradient (DDPG): the first Actor-Critic DRL algorithm.
- Proximal Policy Gradient (PPO): a popular DRL algorithm.


`helloworld` is made simple.
- **Fewer lines of code**. (number of lines < 1000)
- **Little packages requirements**. (`torch` and `gym`)
- **Keep a consistent style with ElegantRL**.


## Run.

You can run the single file of DQN, DDPG and PPO:
- DQN (off-policy DRL algorithm for discrete action space) `helloworld/helloworld_DQN_single_file.py`
- DDPG (off-policy DRL algorithm for continuous action space) `helloworld/helloworld_DDPG_single_file.py`
- PPO (off-policy DRL algorithm for continuous action space) `helloworld/helloworld_PPO_single_file.py`

Or you can:
1. Build the folder `helloworld`. 
2. Put `net.py`, `agent.py`, `config.py`, `env.py`, `run.py` and `tutorial_*.py` in this folder.
3. Run `tutorial_*.py` in this folder. 

In `tutorial_*.py`, there are:
```
train_dqn_for_cartpole()
train_dqn_for_lunar_lander()
train_ddpg_for_pendulum()
train_ppo_for_pendulum()
train_ppo_for_lunar_lander()
```

---

![File_structure of ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL/raw/master/figs/File_structure.png)

One sentence summary: an agent `agent.py` with Actor-Critic networks `net.py` is trained `run.py` by interacting with an environment `env.py`.


---

## The training log

The training log of DQN:
```
| `step`: Number of samples
| `time`: Time spent from the start of training to this moment.
| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode.
| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode.
| `avgS`: Average of steps in an episode.
| `objC`: Objective of Critic network. Or call it loss function of critic network.
| `objA`: Objective of Actor network. It is the average Q value of the critic network.
env_args = {'env_name': 'CartPole-v0',
            'state_dim': 4,
            'action_dim': 2,
            'if_discrete': True}
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.05e+04        60  |     9.38    0.82       9  |     5.79     85.69
| 4.10e+04       182  |   157.41   29.16     157  |     0.73     20.82
| 6.14e+04       320  |   138.72   23.36     139  |     0.31     19.65
| 8.19e+04       471  |   106.91   11.28     107  |     0.21     19.49

env_args = {'env_name': 'LunarLander-v2',
            'state_dim': 8,
            'action_dim': 4,
            'if_discrete': True}
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.05e+04       142  |   -30.49   19.97    1000  |     1.91     20.09
| 8.19e+04       791  |   -27.40   19.82    1000  |     2.30     16.74
| 1.43e+05      1892  |   -54.18  125.79     819  |     2.21     11.73
| 2.05e+05      3480  |   -12.79   70.03     933  |     1.71     15.49
| 2.66e+05      5304  |   167.56  102.91     481  |     1.14     41.37
| 3.28e+05      7443  |   145.19   88.17     664  |     1.18     20.50
| 3.89e+05      9672  |   232.74   35.30     475  |     0.86     18.23
| Save learning curve in ./LunarLander-v2_DQN_0/LearningCurve.jpg
| Press 'y' to load actor.pth and render:y
| render and load actor from: ./LunarLander-v2_DQN_0/actor_000000389120_00009672_00232.74.pth
| DDQN and D3QN train faster than DQN 
```

The training log of DDPG:
```
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.05e+04       108  | -1289.55  159.09     200  |     3.68   -165.12
| 4.10e+04       282  |  -253.97  169.84     200  |     1.81   -162.71
| 6.14e+04       509  |  -150.34   81.19     200  |     1.53    -95.60
```

The training log of PPO:
```
env_args = {'env_name': 'Pendulum-v1',
            'state_dim': 3,
            'action_dim': 1,
            'if_discrete': False}
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 4.00e+04        49  | -1070.47  230.96     200  |    47.91      0.01
| 8.00e+04       106  | -1048.20  124.73     200  |    31.93      0.02
| 1.20e+05       158  |  -841.23   86.72     200  |    15.85      0.01
| 1.60e+05       211  |  -299.09  196.78     200  |    16.56      0.02
| 2.00e+05       263  |  -188.97  127.64     200  |     3.51      0.02
| Save learning curve in ./Pendulum_PPO_0/LearningCurve.jpg

env_args = {'env_name': 'LunarLanderContinuous-v2',
            'state_dim': 8,
            'action_dim': 2,
            'if_discrete': False}
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.00e+04        53  |  -232.54   75.45     197  |    11.75      0.13
| 1.00e+05       689  |   143.02   66.60     828  |     1.91      0.14
| 2.00e+05      1401  |    61.57  133.74     534  |     3.92      0.15
| 3.00e+05      2088  |   108.64  103.73     668  |     2.44      0.18
| 4.00e+05      2724  |   159.55   96.49     522  |     2.37      0.19
| Save learning curve in ./LunarLanderContinuous-v2_PPO_0/LearningCurve.jpg
```

---

# The API of ElegantRL(Helloworld) and ElegantRL

```
run.py 
├── env.py
└── agent.py 
    ├── net.py
    └── config.py
```

- 数据类型 `Tensor` 指代 `torch.Tensor` 
- 数据类型 `Array` 指代 `numpy.ndarray`

## file net.py
### class QNet
DQN系列算法的Q network，，继承自torch默认的网络父类`nn.Module`

`forward(state) -> action` 
- 描述：输出确定策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: shape == (-1, action_dim)`, **各个离散动作的Q值**，格式为`torch.float32`
- 用法：使用策略 `render_agent()`，评估策略`get_rewards_and_steps()`

`get_action(state) -> action`
- 描述：输出随机策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: shape == (-1, 1)`, 对随机策略进行采样后的**离散动作序号**，格式为`torch.int`
- 用法：探索环境 `agent.explore_env()`

### class Actor
Policy gradient 算法的策略网络actor，继承自torch默认的网络父类`nn.Module`

`forward(state) -> action` 
- 描述：输出确定策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: shape == (-1, action_dim)`, 连续动作
- 用法：使用策略 `render_agent()`，评估策略`get_rewards_and_steps()`

`get_action(state) -> action`
- 描述：输出随机策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: shape == (-1, action_dim)`, 对随机策略进行采样后的连续动作
- 用法：探索环境 `agent.explore_env()`

### class Critic
Policy gradient 算法的价值网络critic，继承自torch默认的网络父类`nn.Module`

`forward(state, action) -> q_value` 
- 描述：输出对state-action pairs 的Q值估计
- 输入：
  - `state: shape == (-1, state_dim)` 状态
  - `action: shape == (-1, action_dim)`, 连续动作
- 输出：`q_value: Tensor, q_value.shape == (-1, 1)`, 价值网络对 state-action pairs 的Q值估计
- 用法：计算价值网络的优化目标 `get_obj_critic()`，为策略网络提供梯度 `update_net()`

### class ActorPPO
Policy gradient 算法的策略网络actor，继承自torch默认的网络父类`nn.Module`

`forward(state) -> action` 
- 描述：输出确定策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: shape == (-1, action_dim)`, 连续动作
- 用法：使用策略`render_agent()`，评估策略`get_rewards_and_steps()`

`get_action(state) -> (action, logprob)`
- 描述：输出随机策略
- 输入：`state: shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：
  - `action: shape == (-1, action_dim)`, 对随机策略进行采样后的连续动作
  - `logprob: shape == (-1, )`, **对数概率值 logarithmic probability**，在线策略on-policy 需要这个值去估计随机策略下，当前抽样动作的出现概率
- 用法：探索环境 `agent.explore_env()`

`get_logprob_entropy(state, action) -> (logprob, entropy)`
- 描述：输出随机策略
- 输入：
  - `state: shape == (-1, state_dim)` 状态
  - `action: shape == (-1, action_dim)`, 连续动作
- 输出：
  - `logprob: shape == (-1, )`, **对数概率值 logarithmic probability**，在线策略on-policy 需要这个值去估计随机策略下，当前抽样动作的出现概率
  - `entropy: shape == (-1, )`, **策略的熵**，描述了在当前state-action pairs下，策略的随机程度
- 用法：更新网络参数 `agent.update_net()`

`convert_action_for_env(action) -> action` 
- 描述：将实数范围内的连续动作，处理成有界区间内的连续动作
- 输入：`action: shape == (-1, action_dim)`, 连续动作，范围是整个实数区域`(-inf, +inf)`
- 输出：`action: shape == (-1, action_dim)`, 连续动作，范围是一个有界区间`(-1.0, +1.0)`
- 用法：将PPO的策略网络输出的原始动作，输入到`env.step(action)` 之前，对无界连续动作处理成有界连续动作

### class Critic
Policy gradient 算法的价值网络critic，继承自torch默认的网络父类`nn.Module`

`forward(state, action) -> q_value` 
- 描述：输出对state-action pairs 的Q值估计
- 输入：
  - `state: shape == (-1, state_dim)` 状态
  - `action: shape == (-1, action_dim)`, 连续动作
- 输出：`q_value: Tensor, q_value.shape == (-1, 1)`, 价值网络对 state-action pairs 的Q值估计
- 用法：计算价值网络的优化目标 `get_obj_critic()`，为策略网络提供梯度 `update_net()`

### utils

`build_mlp(dims) -> mlp_net`
- 描述：新建一个MLP网络，默认在两层之间使用激活函数 `ReLU`，输出层后没有激活函数
- 输入：`dims: [int]`。例如`dims=[2, 4, 1]` 新建一个输入特征数量为2，隐藏层为4，输出层为1的全连接网络。
- 输出：`mlp_net: nn.Sequantial` 新建的一个MLP网络
- 用法：只给`net.py` 里的任何类调用（**是否要在这个函数前面加 一个下划线 表示私有？**）


## file agent.py

### class AgentBase
ElegantRL库 所有算法的基类，基类的初始化方法，会记录训练网络需要的超参数，新建神经网络实例，定义网络优化器，定义损失函数

`__init__(self, net_dims, state_dim, action_dim, gpu_id, args)` 
- 描述：初始化一个确定策略算法
- 输入：
  - `net_dims: [int]` 这里输入的只是网络隐藏层的特征数量列表，如 `net_dims=[32, 16]` 将建立的策略网络每层特征数量为 `[state_dim, 32, 16, action_dim]`
  - `state_dim: int` 状态向量的特征数量
  - `action_dim: int` 动作向量的特征数量（或者是离散动作的个数）
  - `gpu_id: int` 表示GPU的编号，用于获取计算设备，`gpu_id=-1`表示用CPU计算。有`torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")`
  - `args: Config()` 记录超参数的类。强化学习的超参数多，我们整理出必要超参数放在 `Config()`这个类里，如果想要RL算法需要用到新增超参数，可以用 `args=Config(); args.xxxx=*` 直接新建超参数，避免频繁修改库的底层文件。所以这里才使用 args 去传递超参数，而不是直接将超参数在 `__init__` 里面展开。（欢迎讨论更好的超参数传输方法）
- 输出：无
- 用法：所有算法的基类，新建的算法 `AgentXXX` 会继承这个基类

`optimizer_update(optimizer, objective)`
- 描述：使用优化器去优化目标
- 输入：
  - `optimizer` 神经网络的优化器。常用的有随机梯度下降+动量 SGD+momentum，收敛快但泛化性能略低的Adam。例如`optimizer = torch.optim.SGD(network.parameters(), learning_rate)` 
  - `objective` 神经网络的优化目标。 神经网络的优化器会根据优化目标提供的梯度，更新网络的参数去最小化优化目标。（价值网络的）优化目标是“最小化预测值与标签的距离”时，可以把这个衡量距离的函数称为“损失函数”。（策略网络的）优化目标是“最大化 价值网络输出的Q值”，等同于“最小化 价值网络输出的Q值的负数”，这种情况下不适合将称之为“损失函数”
- 输出：`mlp_net: nn.Sequantial` 新建的一个MLP网络
- 用法：给`update_net()`用来更新网络参数

`soft_update(target_net, current_net, tau)`
- 描述：使用软更新的方法更新目标网络。可以稳定强化学习的训练过程。可以理解成软更新得到的 `target_net` 是对不同时间截面上的 当前网络 的加权平均。越远离当前时刻的 `current_net` 的权重越低。软更新可以在优化目标含有的噪声较大的情况下，减弱噪声对稳定训练的影响

- 输入：
  - `target_net: nn.module` 需要被更新的目标网络
  - `current_net：nn.module` 用于更新目标网络的当前网络
  - `tau：float` 是一个位于0.0到1.0之间的数值。`tau=1.0`表示直接把`current_net`复制给`target_net`，相当于在“硬更新”。 `tau` 越大更新给`target_net` 带来的更新幅度越大。软更新使用这条公式`target_net = target_net * (1-tau) +current_net * tau`
  - `gpu_id: int` 表示GPU的编号，用于获取计算设备，`gpu_id=-1`表示用CPU计算。有`torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")`
  - `args: Config()` 记录超参数的类。强化学习的超参数多，我们整理出必要超参数放在 `Config()`这个类里，如果想要RL算法需要用到新增超参数，可以用 `args=Config(); args.xxxx=*` 直接新建超参数，避免频繁修改库的底层文件。所以这里才使用 args 去传递超参数，而不是直接将超参数在 `__init__` 里面展开。（欢迎讨论更好的超参数传输方法）
- 输出：无（在原地更新`target_net`，不需要输出值）
- 用法：off-policy算法在更新`current_net`的网络参数后，可以选用这个函数对 `target_net` 进行软更新

### class AgentDQN
算法DQN

`explore_env(self, env, horizon_len, if_random) -> (states, actions, rewards, undones, info_dict)`
- 描述：让智能体在环境中探索，并收集用于训练的数据
- 输入：
  - `env: Object` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
    - `horizon_len: int` 在每一轮探索中，智能体与环境的互动步数。控制了每轮更新中用于网络训练的新增样本数。触发`done=True`时，调用`env.reset()`去重置环境得到`next_state`
- 输出：
  - `states: shape == (horizon_len, state_dim)` 按时刻有序排列的状态state
  - `actions: shape == (horizon_len, 1)` 按时刻有序排列的动作action，**这里将记录离散动作的序号，为torch.int`格式的整数**
  - `rewards: shape == (horizon_len, 1)` 按时刻有序排列的奖励reward
  - `undones: shape == (horizon_len, 1)` 按时刻有序排列的停止标记undone。episode结束时有`undone=False`，其余时刻`undone=True`
  - `info_dict: dict` 记录额外信息的字典。可以传`None`表示没有需要传输的数据。
- 用法：在`run.train_agent()`函数里，让智能体与环境交互获取训练所需的数据`state, action, reward, undone`。

`update_net(self, buffer) -> (obj_critic_value, obj_actor_value, ...)`
- 描述：根据算法设定的优化目标优化网络参数，并输出训练日志。查看训练日志，画出中间变量的曲线，可以给RL训练超参数提供修改思路。
- 输入：
  - `buffer: ReplayBuffer` 经验回放缓存，有更新缓存数据的方法`buffer.update()`，以及随机抽取环境数据的方法`buffer.sample()`  
- 输出：输出一个记录了训练产生的日志数据，组成了一个浮点数元组。
  - `obj_critic_value: float` critic网络的优化目标是：最小化critic拟合Q值的损失函数，这里输出critic的损失函数的均值
  - `obj_actor_value: float` actor网络的优化目标是：最大化critic给actor动作估计的Q值，这里输出的是Q值估计的均值
  - `...: float` 其他数值
- 用法：在`run.train_agent()`函数里，让智能体使用 ReplayBuffer 更新网络

`get_obj_critic(self, buffer, batch_size) -> (obj_critic, states)`
- 描述：先计算得到需要critic去拟合的Q值作为标签，然后输出critic估计的Q值与标签的距离
- 输入：
  - `buffer: ReplayBuffer` 经验回放缓存，有更新缓存数据的方法`buffer.update()`，以及随机抽取环境数据的方法`buffer.sample()`  
  - `batch_size: int` 每一次从经验回放缓存 `buffer` 中随机抽取样本的个数
- 输出：
  - `obj_critic: Tensor` critic网络的优化目标是：最小化critic拟合Q值的损失函数
  - `state: shape == (horizon_len, state_dim)` 按时刻有序排列的状态state
- 用法：在`update_net()`函数里，得到critic网络的优化目标，以及用于计算actor网络优化目标的`states`，**DQN算法的Qnet可以视为拟合Q值的critic网络，DQN没有actor网络**


### class AgentDDPG
算法DDPG

`explore_env(self, env, horizon_len, if_random) -> (states, actions, rewards, undones, info_dict)`
- 描述：让智能体在环境中探索，并收集用于训练的数据
- 输入：
  - `env:` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
    - `horizon_len: int` 在每一轮探索中，智能体与环境的互动步数。控制了每轮更新中用于网络训练的新增样本数。触发`done=True`时，调用`env.reset()`去重置环境得到`next_state`
- 输出：
  - `states: shape == (horizon_len, state_dim)` 按时刻有序排列的状态state
  - `actions: shape == (horizon_len, action_dim)` 按时刻有序排列的动作action
  - `rewards: shape == (horizon_len, 1)` 按时刻有序排列的奖励reward
  - `undones: shape == (horizon_len, 1)` 按时刻有序排列的停止标记undone。episode结束时有`undone=False`，其余时刻`undone=True`
  - `info_dict: dict` 记录额外信息的字典。可以传`None`表示没有需要传输的数据。
- 用法：在`run.train_agent()`函数里，让智能体与环境交互获取训练所需的数据`state, action, reward, undone`。

`update_net(self, buffer) -> (obj_critic_value, obj_actor_value, ...)`
- 描述：根据算法设定的优化目标优化网络参数，并输出训练日志。查看训练日志，画出中间变量的曲线，可以给RL训练超参数提供修改思路。
- 输入：
  - `buffer: ReplayBuffer` 经验回放缓存，有更新缓存数据的方法`buffer.update()`，以及随机抽取环境数据的方法`buffer.sample()`  
- 输出：输出一个记录了训练产生的日志数据，组成了一个浮点数元组。
  - `obj_critic_value: float` critic网络的优化目标是：最小化critic拟合Q值的损失函数，这里输出critic的损失函数的均值
  - `obj_actor_value: float` actor网络的优化目标是：最大化critic给actor动作估计的Q值，这里输出的是Q值估计的均值
  - `...: float` 其他数值
- 用法：在`run.train_agent()`函数里，让智能体使用 ReplayBuffer 更新网络

`get_obj_critic(self, buffer, batch_size) -> (obj_critic, states)`
- 描述：先计算得到需要critic去拟合的Q值作为标签，然后输出critic估计的Q值与标签的距离
- 输入：
  - `buffer: ReplayBuffer` 经验回放缓存，有更新缓存数据的方法`buffer.update()`，以及随机抽取环境数据的方法`buffer.sample()`  
  - `batch_size: int` 每一次从经验回放缓存 `buffer` 中随机抽取样本的个数
- 输出：
  - `obj_critic: Tensor` critic网络的优化目标是：最小化critic拟合Q值的损失函数
  - `state: shape == (horizon_len, state_dim)` 按时刻有序排列的状态state
- 用法：在`update_net()`函数里，得到critic网络的优化目标，以及用于计算actor网络优化目标的`states`

### class AgentPPO
算法PPO

`explore_env(self, env, horizon_len, if_random) -> (states, actions, logprobs, rewards, undones, info_dict)`
- 描述：让智能体在环境中探索，并收集用于训练的数据
- 输入：
  - `env:` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
    - `horizon_len: int` 在每一轮探索中，智能体与环境的互动步数。控制了每轮更新中用于网络训练的新增样本数。触发`done=True`时，调用`env.reset()`去重置环境得到`next_state`
- 输出：
  - `states: shape == (horizon_len, state_dim)` 按时刻有序排列的状态state
  - `actions: shape == (horizon_len, action_dim)` 按时刻有序排列的动作action
  - `logprobs: shape == (horizon_len, )` 按时刻有序排列的**对数概率值 logarithmic probability**，在线策略on-policy 需要这个值去估计随机策略下，当前抽样动作的出现概率
  - `rewards: shape == (horizon_len, 1)` 按时刻有序排列的奖励reward
  - `undones: shape == (horizon_len, 1)` 按时刻有序排列的停止标记undone。episode结束时有`undone=False`，其余时刻`undone=True`
  - `info_dict: dict` 记录额外信息的字典。可以传`None`表示没有需要传输的数据。
- 用法：在`run.train_agent()`函数里，让智能体与环境交互获取训练所需的数据`state, action, reward, undone`。

`update_net(self, buffer) -> (obj_critic_value, obj_actor_value, ...)`
- 描述：根据算法设定的优化目标优化网络参数，并输出训练日志。查看训练日志，画出中间变量的曲线，可以给RL训练超参数提供修改思路。
- 输入：
  - `buffer: ReplayBuffer` 经验回放缓存，有更新缓存数据的方法`buffer.update()`，以及随机抽取环境数据的方法`buffer.sample()`  
- 输出：输出一个记录了训练产生的日志数据，组成了一个浮点数元组。
  - `obj_critic_value: float` critic网络的优化目标是：最小化critic拟合Q值的损失函数，这里输出critic的损失函数的均值
  - `obj_actor_value: float` actor网络的优化目标是：最大化critic给actor动作估计的Q值，这里输出的是Q值估计的均值
  - `...: float` 其他数值
- 用法：在`run.train_agent()`函数里，让智能体使用 ReplayBuffer 更新网络

`get_advantages(self, rewards, undones, values) -> (obj_critic, states)`
- **描述：计算在线策略的优势值**
- 输入：
  - `rewards: shape == (horizon_len, 1)` 按时刻有序排列的奖励reward
  - `undones: shape == (horizon_len, 1)` 按时刻有序排列的停止标记undone。episode结束时有`undone=False`，其余时刻`undone=True`
  - `values: shape == (horizon_len, )` **按时刻有序排列的优势值估计，不带梯度。由在线策略的critic网络(advantage value function)，基于旧的策略算出**
- 输出：
  - `advantages: shape == (horizon_len, )` **按时刻有序排列的优势值估计，带有梯度。由在线策略的critic网络(advantage value function)，基于新的策略算出**
- 用法：在`update_net()`函数里，**算出即将被优化的策略的优势值**，用于后续计算actor网络的优化目标

---

## 文件 config.py
存放超参数配置，用内置函数获取标准gym仿真环境参数，根据超参数和方法新建仿真环境

### class Config
配置超参数

`__init__(self, agent_class, env_class, env_args)`
- 描述：初始化配置超参数的类。初始化时，除了一些超参数有默认值，它还会自动从`agent_class`获取训练所需要的超参数。从`env_args`获取构建智能体所需的超参数。
- 输入：
  - `agent_class: AgentBase` 存放在 `agent.py` 里，继承自父类 `AgentBase` 的子类们 `AgentXXX`
  - `env_class: gym.Wrapper or Any` 仿真环境的类，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
  - `env_args: dict` 仿真环境的超参数字典，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
- 用法：继承自父类 `AgentBase` 的子类们 `AgentXXX` 都需要读取 `Config`这个类获取，获取初始化所需要的超参数。在`run.py`里，需要先创建 `args=Config(...)`用于记录自定义的超参数

`init_before_training(self)`
- 描述：在训练前，需要使用这个函数设置随机种子，并自动设置工作目录路径，并根据 `if_remove` 超参数决定历史数据是否保留。
- 用法：使用 `run.py`里的 `train_agent()` 函数开始训练前，需要调用这个函数自动配置工作目录`cwd`。

`get_if_off_policy(self) -> if_off_policy`
- 描述：根据`self.agent_class` 的名字，判断这个算法是否是`off-policy`算法
  - 输出：
    - `if_off_policy: bool` 当算法名字含有 SARSA, VPG, A2C, A3C, TRPO, PPO, MPO 时，输出 `False`
- 用法：
  - 在初始化方法`__init__` 的内部，需要知晓这个算法是 `Off-policy` 还是 `On-policy`，用于自动匹配对应的超参数
  - 使用 `run.py`里的 `train_agent()` 函数进行训练时，需要判断这个算法是 `Off-policy` 还是 `On-policy`，才能选择合适的`ReplayBuffer`用于匹配训练模式。

### utils
仿真环境的超参数获取与使用

`get_gym_env_args(env, if_print) -> env_args`
- 描述：获取标准gym环境的超参数，用于初始化强化学习智能体
- 输入：
  - `env: Object` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
  - `if_print: bool` 是否把打印环境的超参数`env_args`。打印出来后，可以直接写在配置文件里，就不需要再调用这个函数重新获取环境的超参数字典。
- 输出：
  - `env_args: dict` 这个环境超参数字典包含了：仿真环境的名字 `env_name`，任务的状态数量`state_dim`，动作数量`action_dim`，以及动作空间是否为离散`if_discrete`
- 用法：在使用一个符合gym的标准仿真环境前，可以使用这个函数自动获取环境的超参数字典，

`kwwargs_filter(function, kwargs) -> env_args`
- 描述：使用仿真环境的超参数新建环境前，需要对超参数进行过滤，用以匹配仿真环境的类的初始化超参数。 KeyWordArgumentFilter
- 输入：
  - `function: Object` 仿真环境的类的初始化函数`env_class.__init__`，可以直接输入仿真环境的类 `env_class`
  - `kwargs: dict` 需要进行过滤的环境超参数 `env_args`
- 输出；
  - `env_args: dict` 过滤后的环境超参数
- 用法：新建环境时，调用 `build_env()` 函数，里面就需要过滤 `env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))`

`build_env(env_class, env_args) -> env`
- 描述：通过仿真环境的类，以及仿真环境的超参数字典，新建仿真环境实例。注意，当超过`0.25.2`的gym版本，`env.reset()`函数的返回不是 `state` 而是 `state, info_dict`，会有兼容问题。
- 输入：
  - `env_class: gym.Wrapper or Any` 仿真环境的类，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
  - `env_args: dict` 仿真环境的超参数字典，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
- 输出：
  - `env: Object` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
- 用法：使用 `run.py`里的 `train_agent()` 函数进行训练时，需要新建一个仿真环境用于与智能体交互，新建另一个仿真环境环境用于评估智能体表现

## 文件 env.py
存放了自定义环境

### class PendulumEnv
一个基于 OpenAI gym 倒立摆 Pendulum 的自定义环境

`__init__(self)`
- 描述：自定义仿真环境的初始化函数，它会检查gym版本不超过`0.25.2`。然后主动设置仿真环境的信息，包括：仿真环境的名字 `env_name`，任务的状态数量`state_dim`，动作数量`action_dim`，以及动作空间是否为离散`if_discrete`
- 用法：建环境时，调用 `build_env()` 函数，初始化函数被自动调用

`reset() -> state`
- 描述：仿真环境里，通过 `env.reset()` 将仿真环境进行重置，重置后智能体将从轨迹`trajectory`的开头重新探索。
  - 输出：
    - `state` 仿真环境首个时刻的状态。设计自己的仿真环境时，最好让第一个时刻的状态带有足够的随机性，这能提高智能体的泛化性能
- 用法：在`agent.py`里，继承自父类 `AgentBase` 的子类们 `AgentXXX` 都需要使用`env.reset()`
  - 在初始化后，使用 `agent.last_state = env.reset()` 对 `last_state` 进行初始化，用于探索环境时，根据上一个 `state` 选择对应的 `action` 
  - 在探索环境阶段，使用`agent.explore_env()`，当智能体在仿真环境中触发`done=True`之后，需要重置仿真环境。并行仿真环境会自动调用子环境的`env.reset()`方法

`step(self, action) -> next_state, reward, done, info_dict`
- 描述：仿真环境的交互函数，输入智能体的动作，得到仿真环境的反馈
- 输入：
  - `state` 仿真环境某个时刻的状态。与动作`action` 是同一时刻
- 输出：
  - `next_state: Array` 仿真环境下个个时刻的状态。是动作`action` 的下一时刻
  - `reward: float` 仿真环境某个时刻的奖励。与动作`action` 是同一时刻
  - `done: bool` 仿真环境某个时刻的轨迹终止信号。与动作`action` 是同一时刻。`done=True` 表示这个轨迹停止了，之后的轨迹与之前的轨迹相互独立。
  - `info_dict: dict` 仿真环境的其他输出，都可以放在此字典内。
- 用法：在`agent.py`里
  - 继承自父类 `AgentBase` 的子类们 `AgentXXX` 都需要在探索环境阶段`agent.explore_env()`中，使用`env.step(...)`让智能体与环境交互得到训练数据
  - 评估智能体表现的 `Evaluator` 都需要在评估阶段 `evaluator.evaluate_and_save()` 中，使用`env.step(...)`让智能体与环境交互得到智能体表现

## 文件 run.py
强化学习的训练流程

### utils
训练并评估智能体

`train_agent(args)`
- 描述：训练强化学习智能体，并顺带用 Evaluator 评估智能体的表现，输出训练日志
- 输入：
  - `args: Config` 训练需要的超参数，写在 `config.py`文件的 `class Config` 里
- 用法：用户使用 `Config`类 配置好超参数后，直接传入这个函数，它将会调用其他文件完成训练。

`render_agent(env_class, env_args, net_dims, agent_class, actor_path, render_times)`
- 描述：渲染出强化学习智能体的实际表现（需要仿真环境带有 `env.render()` 方法才能调用），让用户能直观看到智能体的表现，并在终端打印出每个episode的累积收益以及步数。
- 输入：
  - `env_class: gym.Wrapper or Any` 仿真环境的类，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
  - `env_args: dict` 仿真环境的超参数字典，新建仿真环境会用到这段代码：`env = env_class(**env_args)`
  - `net_dims: [int]` 这里输入的只是网络隐藏层的特征数量列表，如 `net_dims=[32, 16]` 将建立的策略网络每层特征数量为 `[state_dim, 32, 16, action_dim]`
  - `agent_class: AgentBase` 存放在 `agent.py` 里，继承自父类 `AgentBase` 的子类们 `AgentXXX`，这里需要创建`agent=agent_class(...)`，然后使用策略网络 `agent.act`
  - `actor_path: str` 保存了策略网络模型参数的路径，一般是 `f"{cwd}/actor_**.pth"` 
  - `render_times: int` 渲染的轮数。也就是触发 `done=True`的次数。
- 用法：用户使用 `train_agent()`得到训练好的模型文件后，可以调用这个函数直观感受智能体的实际表现。



### class Evaluator
评估智能体，记录并展示训练日志

`__init__(self, eval_env, eval_per_step, eval_times, cwd)`
- 描述：评估器Evaluator 的初始化函数，它会打印训练日志每一项的解释，并准备输出和记录训练日志
- 输入：
  - `eval_env: Obejct` 用于评估智能体的仿真环境。因为训练用的仿真环境需要保存上一次探索的状态，不能随便调用`env.reset()`。所以评估的环境不与训练的环境共用
  - `eval_per_step: int` 每隔固定步数，对智能体进行一轮评估，得到智能体的 平均累积奖励、每轮平均步数
  - `eval_times: int` 对智能体进行一轮评估的时候，运行固定次数的episodes，用来看智能体的表现是否稳定，得到对智能体更加准确的评估
  - `cwd: str` 当前的工作目录 current working directory。用来保存训练日志`recorder.npy` 以及学习曲线 `LearningCurve.jpg`
- 用法：用户使用`train_agent()`训练智能体开始之前，需要新建一个 评估器 Evaluator 的实例，用于之后记录训练日志，帮助超参数调优，训练出更好表现的智能体。

`evaluate_and_save(self, actor, horizon_len, logging_tuple)`
- 描述：评估智能体，并保存训练日志
- 输入：
  - `actor: nn.Module` 智能体的策略网络，输入当前时刻的state，它能映射得到当前时刻策略采取的动作action
  - `horizon_len: int` 在每一轮探索中，智能体与环境的互动总步数。记录这个值，得到每个时刻的训练总步数`total_step`，在绘制学习曲线时，可以使用训练步数作为横坐标衡量训练量
  - `logging_tuple: (float)` 输出训练产生的中间变量，作为训练日志。`logging_tuple = objC, objA, ...`。在评估器Evaluator的初始化方法中，有详细的解释
- 用法： 用户使用`train_agent()`训练智能体的同时，需要同时调用 评估器 Evaluator 记录训练日志，帮助超参数调优，训练出更好表现的智能体。

`save_training_curve_jpg(self)`
- 描述：保存训练日志，并根据训练日志画出训练曲线（包括学习曲线，以及其他训练中间变量的曲线），保存在`f"{cwd}/LearningCurve.jpg"`
- 用法：评估器Evaluator 可以每隔一段时间调用这个函数画出训练曲线，也可以在训练过程中间隔固定时间实时画出图片

### utils of class Evaluator
评估器会用到的函数

`get_rewards_and_step(env, actor, if_render) -> cumulative_rewards, step`
- 描述：得到智能体的累积奖励，以及当前episode的步数，用于评估智能体表现
- 输入：
  - `env: Object` 用于智能体训练的仿真环境，含有两个方法：重置环境`env.reset()`，与智能体互动`env.step()`
  - `actor: nn.Module` 智能体的策略网络，输入当前时刻的state，它能映射得到当前时刻策略采取的动作action
  - `if_render: bool` 是否渲染出智能体与环境互动的画面。
- 输出；
  - `cumulative_rewards: float` 智能体与环境互动，得到的一个episode内所有奖励的求和，称为累积奖励求和
  - `step: int` 智能体与环境互动的一个episode内的步数，从 `env.reset()` 开始算一步，到`done=True`的时候停止计数。
- 用法：
  - 评估器 Evaluator 在记录训练日志时`evaluator.evaluate_and_save()`，令`if_render=False`，只调用这个函数得到智能体的 平均累积奖励 以及 每轮平均步数。 
  - 用户使用`render_agent()`渲染出智能体与环境互动的画面，会令`if_render=True`，直观感受智能体实际表现。 

`draw_learning_curve(recorder, save_path)`
- 描述：根据训练日志文件`recorder` 以及训练日志的保存路径 `save_path` 画出训练曲线（包括学习曲线，以及其他训练中间变量的曲线）
- 输入：
  - `recorder: Array` 训练日志文件，记录了智能体在达到某个采样步数时，它的训练耗时，以及平均累积奖励，可以在 `evaluator.evaluate_and_save()` 里面搜索`self.recorder.append((self.total_step, used_time, avg_r))` 看看我们往里面装了什么数据
  - `save_path: str` 训练曲线画成图片后的保存路径，默认是`f"{cwd}/LearningCurve.jpg"`

