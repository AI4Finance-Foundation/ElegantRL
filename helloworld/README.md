## ElegantRL helloworld 

Three algorithms are included: 
- Deep Q-Network (DQN): the first DRL algorithm.
- Deep Deterministic Policy Gradient (DDPG): the first Actor-Critic DRL algorithm.
- Proximal Policy Gradient (PPO): a popular DRL algorithm.


`helloworld` is made simple.
- **Less lines of code**. (code lines <1000)
- **Little packages requirements**. (`torch` and `gym`)
- **Keep a consistent style with ElegantRL**.


## Run.

You can run the single file of RL algorithms DQN, DDPG and PPO:
- DQN (off-policy RL algorithm for discrete action space) `helloworld/helloworld_DQN_single_file.py`
- DDPG (off-policy RL algorithm for continuous action space) `helloworld/helloworld_DDPG_single_file.py`
- PPO (off-policy RL algorithm for continuous action space) `helloworld/helloworld_PPO_single_file.py`

Or you can:
1. Build the folder `helloworld` in the current working directory. 
2. Put `net.py`, `agent.py`, `config.py`, `env.py`, `run.py` and `tutorial_*.py` in the folder `helloworld`.
3. Run `tutorial_*.py`. 

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
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.05e+04        48  |     9.25    0.66       9  |     6.13     90.66
| 4.10e+04       120  |   129.84   22.61     130  |     0.81     21.17
...
| 1.64e+05      1122  |   115.88   15.86     116  |     0.18     19.58
| 1.84e+05      1414  |   152.72   30.05     153  |     0.17     19.42
```

The training logging of DDPG:
```
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.05e+04        83  | -1257.86   95.30     200  |     2.87   -155.83
| 4.10e+04       224  |  -980.77   65.55     200  |     2.36   -168.30
| 6.14e+04       417  |  -838.92  140.21     200  |     1.94   -153.69
| 8.19e+04       666  |  -245.80  263.86     200  |     2.38   -134.42
```

The training logging of PPO:
```
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.00e+04        22  | -1335.30  239.78     200  |    56.31      0.02
| 4.00e+04        45  | -1311.17  227.48     200  |    44.50      0.02
...
| 1.80e+05       203  |  -174.62  125.42     200  |     3.04      0.02
| 2.00e+05       225  |  -255.44  277.14     200  |     3.74      0.02
```

---

# The API of ElegantRL(Helloworld) and ElegantRL

```
run.py 
├── env.py
└── agent.py 
    ├── net.py
    └── config.py
net.py    -->  agent.py
config.py -->  agent.py
```
- net.py -> agent.py
- (config.py, env.py) -> run.py

- 数据类型 `Tensor` 指代 `torch.Tensor` 
- 数据类型 `Array` 指代 `numpy.ndarray`

## file net.py
### class Qnet
DQN系列算法的Q network，，继承自torch默认的网络父类`nn.Module`

`forward(state) -> action` 
- 描述：输出确定策略
- 输入：`state: Tensor, state.shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: Tensor, action.shape == (-1, action_dim)`, **各个离散动作的Q值**，格式为`torch.float32`
- 用法：使用策略 `render_agent()`，评估策略`get_rewards_and_steps()`

`get_action(state) -> action`
- 描述：输出随机策略
- 输入：`state: torch.float, state.shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: torch.int, action.shape == (-1, 1)`, 对随机策略进行采样后的**离散动作序号**，格式为`torch.int`
- 用法：探索环境 `agent.explore_net()`

### class Actor
Policy gradient 算法的策略网络actor，继承自torch默认的网络父类`nn.Module`

`forward(state) -> action` 
- 描述：输出确定策略
- 输入：`state: Tensor, state.shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: Tensor, action.shape == (-1, action_dim)`, 连续动作
- 用法：使用策略 `render_agent()`，评估策略`get_rewards_and_steps()`

`get_action(state) -> action`
- 描述：输出随机策略
- 输入：`state: torch.float, state.shape == (-1, state_dim)`, 策略当前时刻的状态
- 输出：`action: torch.int, action.shape == (-1, action_dim)`, 对随机策略进行采样后的连续动作
- 用法：探索环境 `agent.explore_net()`

### class Critic
Policy gradient 算法的价值网络critic，继承自torch默认的网络父类`nn.Module`

`forward(state, action) -> q_value` 
- 描述：输出对state-action pairs 的Q值估计
- 输入：`state: Tensor action: Tensor` state-action pairs
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

`__init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config())` 
- 描述：输出确定策略
- 输入：
  - `net_dims: [int]` 这里输入的只是网络隐藏层的特征数量列表，如 `net_dims=[32, 16]` 将建立的策略网络每层特征数量为 `[state_dim, 32, 16, action_dim]`
  - `state_dim: int` 状态向量的特征数量
  - `action_dim: int` 动作向量的特征数量（或者是离散动作的个数）
  - `gpu_id: int` 表示GPU的编号，用于获取计算设备，`gpu_id=-1`表示用CPU计算。有`torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")`
  - `args: Config()` 记录超参数的类。强化学习的超参数多，我们整理出必要超参数放在 `Config()`这个类里，如果想要RL算法需要用到新增超参数，可以用 `args=Config(); args.xxxx=*` 直接新建超参数，避免频繁修改库的底层文件。所以这里才使用 args 去传递超参数，而不是直接将超参数在 `__init__` 里面展开。（欢迎讨论更好的超参数传输方法）
- 输出：无
- 用法：ElegantRL库 所有算法的基类，新建的算法 `AgentXXX` 会继承这个基类

`optimizer_update(optimizer, objective)`
- 描述：使用优化器去优化目标
- 输入：
  - `optimizer` 神经网络的优化器。常用的有随机梯度下降+动量 SGD+momentum，收敛快但泛化性能略低的Adam。例如`optimizer = torch.optim.SGD(network.parameters(), learning_rate)` 
  - `objective` 神经网络的优化目标。 神经网络的优化器会根据优化目标提供的梯度，更新网络的参数去最小化优化目标。（价值网络的）优化目标是“最小化预测值与标签的距离”时，可以把这个衡量距离的函数称为“损失函数”。（策略网络的）优化目标是“最大化 价值网络输出的Q值”，等同于“最小化 价值网络输出的Q值的负数”，这种情况下不适合将称之为“损失函数”
- 输出：`mlp_net: nn.Sequantial` 新建的一个MLP网络
- 用法：只给`net.py` 里的任何类调用（**是否要在这个函数前面加 一个下划线 表示私有？**）

`soft_update(target_net, current_net, tau)`
- 描述：使用软更新的方法更新目标网络。可以稳定强化学习的训练过程。可以理解成软更新得到的 `target_net` 是对不同时间截面上的 当前网络 的加权平均。越远离当前时刻的 `current_net` 的权重越低。软更新可以在优化目标含有的噪声较大的情况下，减弱噪声对稳定训练的影响

- 输入：
  - `target_net: nn.module` 需要被更新的目标网络
  - `current_net： nn.module` 用于更新目标网络的当前网络
  - `tau：float` 是一个位于0.0到1.0之间的数值。`tau=1.0`表示直接把`current_net`复制给`target_net`，相当于在“硬更新”。 `tau` 越大更新给`target_net` 带来的更新幅度越大。软更新使用这条公式`target_net = target_net * (1-tau) +current_net * tau`
  - `gpu_id: int` 表示GPU的编号，用于获取计算设备，`gpu_id=-1`表示用CPU计算。有`torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")`
  - `args: Config()` 记录超参数的类。强化学习的超参数多，我们整理出必要超参数放在 `Config()`这个类里，如果想要RL算法需要用到新增超参数，可以用 `args=Config(); args.xxxx=*` 直接新建超参数，避免频繁修改库的底层文件。所以这里才使用 args 去传递超参数，而不是直接将超参数在 `__init__` 里面展开。（欢迎讨论更好的超参数传输方法）
- 输出：无（在原地更新`target_net`，不需要输出值）
- 用法：off-policy算法在更新`current_net`的网络参数后，可以选用这个函数对 `target_net` 进行软更新

### class AgentDQN
算法DQN

**todo 描述 DQN算法以及它的变体相对于 AgentBase 的差别**

`explore_env(env, horizon_len, if_random) -> buffer_items`
- 描述：探索环境
- 输入：
- 输出：
- 用法：

`update_net(buffer) -> training_logging`
- 描述：根据算法设定的优化目标优化网络参数，并输出训练日志。查看训练日志，画出中间变量的曲线，可以给RL训练超参数提供修改思路。
- 输入：
- 输出：
- 用法：


### class AgentDDPG
算法DDPG

**todo 描述 off-policy (continuous action) DRL算法相对于 AgentBase 的差别**

### class AgentPPO
算法PPO

**todo 描述 on-policy DRL算法相对于 off-policy DRL算法的差别**

---

## 文件 config.py
## 文件 env.py
## 文件 run.py
