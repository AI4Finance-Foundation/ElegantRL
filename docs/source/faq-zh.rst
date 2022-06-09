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

有空我就展开讲一讲。把下面这段我写的蹩脚英文改回中文。


^^^^^^^^
Q1: In what conditions on-policy is better, and in what conditions off-policy is better in general?
^^^^^^^^

This is a very difficult question to answer, but I will try to answer it.

Let us discuss which kind of algorithm is better under 3 performance metrices:
- Off-policy is better in **sample efficiency** than on-policy in general. (The agent achieves higher cumulative rewards in a given total training steps.). 
- On-policy performs better in **training speed** than off-policy in general. (The agent achieves higher cumulative rewards in a given total training time.).
- On-policy performs better in **training stablility** than off-policy in general. (Train agent overs serval runs after the learning curve converges. A higher training stability algorithm has a smaller the variance of cumulative rewards)
- **Convergence of cumulative rewards** (Train the agent and compare the highest cumulative rewards after the learning curve converges). 
  - Off-policy performs better in **convergence of cumulative rewards** if we can provide sufficient running memoery.
  - On-policy performs better in **convergence of cumulative rewards** if we can not provide sufficient running memoery.


Background about on-policy and off-policy:
- Behavior policy: collects data for training and explores in the environment.
- Target policy: used to update the Q value.
- On-policy algorithm: the target policy must be the behavior policy. So the training data in experimence replay buffer should be collected by behavior policy.
- Off-policy algorithm: the target policy can be any policies. So the training data in experimence replay buffer could be collected by any policies.

-----------------
Sample efficiency:
-----------------
If we focus on sample efficiency as a performance metric, off-policy is better that on-policy in general.

On-policy algorithms use the data collected by the behavior policy to update the target policy. The on-policy algorithms keep updating the target policy, until the difference between the behavior policy and target policy is so large that they cannot be considered as the same policy. 

After updating the target policy, the on-policy algorithms **delete the old training data in replay buffer**, and re-collect the data and treat the latest target policy as the behavior policy.

The target policy of off-policy algorithms can be any policies. So the off-policy algorithm **do not need to delete the old training data** unless the data in experimence replay buffer is too much and the capacity limit is reached. 

So off-policy has higher sample efficiency than on-policy in general. Some RL tasks (e.g. atari game) that require sufficient exploration in order to find a policy with higher cumulative rewards. For such tasks, the off-policy algorithms can achieve better performance with higher sample efficiency, because off-policy algorithms maintain a larger expermience replay buffer than on-policy algorithms.

-----------------
Training speed:
-----------------

A typical training pipeline of RL:
1. Behavior policy explores in environment and collects the data for the experimence replay buffer.
2. Using the data in experimence replay buffer to update the target policy and value network.
3. Remove the old data from the experience replay buffer according to the requirements of the algorithms.
4. Repeat step 1 to step 3 until the training stops.

Off-policy will maintain a larger expermience replay buffer (training set) than on-policy.
In step 2, the off-policy algorithms will training its networks in a larger training set. So the off-policy algorithm take longer to train the networks, which slows down its training speed.

When the time consumed in the step 1 is relatively short (i.e., the training environment runs fast enough), the disadvantage of low sample efficiency of on-policy will be non-obvious. And the step 2 of the on-policy algorithm is shorter than the off-policy. Ultimately, the training speed of on-policy is significantly faster when using the same computing device.

-----------------
Training stability
-----------------

On-policy performs better in **training stablility** than off-policy in general. There are 2 reasons:
- The value network of on-policy just need to predict the Q value of behavior policy, which is easier than off-poicy value network predict the Q value of any policy.
- The behavior policy network of on-policy explores in environment and collect the data for the experimence replay buffer. And the target network is same as the behavior policy. The on-policy algorithm searches for new policies in the neighborhood of behavior policy, so on-policy training is more stable than off-policy because the difference between behavior policy and target policy is smaller.

-----------------
Convergence of cumulative rewards:
-----------------

We can train the agent and compare the cumulative rewards after the learning curve converges. If an algorithm searches for a policy with higher cumulative rewards, we said that it is better.

The off-policy algorithm search for its policy using more data, because off-policy will maintain a larger experimence replay buffer than on-policy. So the off-policy algorithm is more likely to jump out of the local optimum that the on-policy cannot jump out of.

In this case, **off-policy performs better** and get a higher convergence cumulative rewards.

In practice, we cannot provide a large enough experience replay buffer for training because the real-world constraints such as memory. 
- The environment or hehavior policy is so stochastic that it required a considerable amount of experimence replay buffer to hold these data.
- The experimence replay buffer will always hold duplicate data and take up valuable memory space, and the cache space cannot be maximally utilized.

Off-policy will maintain a larger expermience replay buffer (training set) than on-policy. 
In other words, the on-policy algorithm is able to use less running memory to solve the same RL task than the off-policy algorithm. If we cannot provide sufficient running memory for the experimence replay buffer, the on-policy algorithm instead obtains a better convergence score than the off-policy algorithm. 

In this case, **on-policy performs better** and get a higher convergence cumulative rewards.

NOTICE: When we training a DEEP reinforcement learning algorithm, We need running memory to store the data of experimence replay buffer, where memory means the memory of a single GPU and not the memory plugged into the motherboard for the CPU (RAM).

In theory, of course, it is possible to temporarily store data from GPU memory into memory on the motherboard for the CPU, or even use the CPU to train neural networks, but that would be very slow.






^^^^^^^^^^^^
Q2: Is it possible to design an off-policy actor-critic algorithm with only the state input (no action input)? If not, can you explain why?
^^^^^^^^^^^^

It is impossible.

Background knowledge about on-policy and off-policy:
- **Behavior policy**: The policy which **explored in the environment and collected data** for training is behavior policy.
- **Target policy**: The policy which used to **update the Q value** is target policy.
- **On-policy algorithm**: the target policy must be the behavior policy. So the training data in experimence replay buffer should be **collected by behavior policy**.
- **Off-policy algorithm**: the target policy can be any policies. So the training data in experimence replay buffer could be **collected by any policies**.
  
Let's compare the critic network of these algorithms:
- The critic network (value network) **estimates the Q value of the policy**. 
- The critic network of on-policy algorithms (state value network) **estimates the Q value of the behavior policy** using the data collected by behavior policy.
- The critic network of off-policy algorithms (state-action value network) **estimates the Q value of the any policy** using the data collected by different policies.

Why the critic network of off-policy algorithms (state-action value network) estimates the Q value of the any policy but state value network can not do this?

**Because the information of behavior policy can be send to state-action value network via the `action` input.**
By comparison, the state value network can only estimate the Q value of behavior policy, so we can not disign an off-policy algorithm with only the state input (no action input).




	
  
  
  
