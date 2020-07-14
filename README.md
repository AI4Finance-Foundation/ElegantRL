## An lightweight, stable, efficient DRL PyTorch implement
Many people suggested that I could call it "3-Python-file-RL (PyTorch)"

Model-free Deep Reinforcement Learning (DRL) Algorithm: 
+ DDPG, TD3(GAE), PPO, SAC, InterAC, InterSAC (I don't add A2C and TRPO because they are no good enough.)
+ DQN ( DRL), DoubleDQN, GAE for discrete action.
+ More DQN (wait for adding Rainbow, Ape-X, SAC for discrete)
+ Multi-agent (wait for adding MADDPG, COMA, QMIX, QTRAN etc.)

It is a beta version. The part of single-agent DRL has been finished.
After I finish the part of multi-agent, I will officially release it. 
Maybe you can find Multi-agent in folder 'BetaWarning'.

I am still learning English. 我的英文不好，还望包涵。
More details see ↓ (write in Chinese)

更多详细的内容见知乎 [「何须天授？彼可取而代也」用于入门的强化学习库 model-free RL in PyTorch](https://zhuanlan.zhihu.com/p/127792558)

# Show results

You can see these gif/png ↓ in file "Result".

LunarLanderContinuous-v2:
![LunarLanderTwinDelay3](https://github.com/Yonv1943/LightWeight_Stable_ReinfLearning/blob/master/Result/LunarLanderTwinDelay3.gif)
BipedalWalkerHardcore-v2: 
![BipedalWalkerHardcore-v2-total](https://github.com/Yonv1943/DL_RL_Zoo/blob/master/Result/BipedalWalkerHardcore-v2-total-668kb.gif)

You can also see the video in [bilibili Faster Pass BipedalWalkerHardcore-v2 total reward 310](https://www.bilibili.com/video/BV1wi4y187tC). (Mainland China)

For example, the training log is the following two figure.

+ The first is an luck result of BasicAC: `plot_0072E_22405T_701s.png`.
+ The second is an slow result of BasicAC: `plot_0249E_159736T_1867s.png`. 
+ The name of the log figure is `plot_<Episode>_<TotalStep>_<Second>UsedTime.png` when agent reaches the default target reward.

Although BasicAC is my improved DDPG, it is still not stable enough. You can try other DRL algorithm, such as SAC and InterSAC. There are more stable than other algorithm.

<p float="left">
  <img src="https://github.com/Yonv1943/LightWeight_Stable_ReinfLearning/blob/master/Result/BasicAC_LunarLanderContinuous-v2_luck/plot_0072E_22405T_701s.png" width="400" />
  <img src="https://github.com/Yonv1943/LightWeight_Stable_ReinfLearning/blob/master/Result/BasicAC_LunarLanderContinuous-v2_unluck/plot_0249E_159736T_1867s.png" width="400" /> 
</p>

In the above two figure. 
+ Blue curve: Total reward (with explore noise)
+ Red curve: Total reward (without explore noise, mean and std)
+ Grey area: The step of one episode while exploring.
+ Green curve: 'loss' value of actor (mean of Q value estimate)
+ Green area: loss value of critic



You can see more training log in file "Result". 

The following Readme in English is written for Necip and other people who can't read Chinese.

# Requirement

    Necessary:
    | Python 3.7      | Python 3.5+ is ok, because I use Python's multiprocessing     
    | PyTorch 1.0.2   | PyTorch 1.X is ok. After all, PyTorch is not TensorFlow.      

    Not necessary:
    | Numpy 1.19.0    | Numpy 1.X is ok. Numpy is one of the requirement for PyTorch. 
    | gym 0.17.2      | In early gym version, some task XXX-v3 would be XXX-v2        
    | box2d-py 2.3.8  | This box2d-py is OpenAI's instead of the original author's    
    | matplotlib 3.2  | Just for drawing plot
This is why I said it **lightweight**.

# File Structure
    -----file----
    AgentRun.py  # Choose your DRL agent here. Then run this python file for training.
    AgentZoo.py  # There are many model-free DRL algorithms here. You can learn these algorithm here.
    AgentNet.py  # The neural network architecture for actor and critic is here. 
    ----folder---
    BetaWarning  # I put the latest version here. You can find lots of bug here.
    History      # I put the historical files here. You can find the bug-free code here.
There are only 3 python file in my DRL PyTorch implement. (I will add more detail later.)

# run
    python3 AgentRun.py
    # You can see run__demo(gpu_id=0, cwd='AC_BasicAC') in AgentRun.py.
+ In default, it will train a stable-DDPG in LunarLanderContinuous-v2 for 2000 second.
+ It would choose CPU or GPU automatically. Don't worry, I never use `.cuda()`.
+ It would save the log and model parameters file in Current Working Directory `cwd='AC_BasicAC'`. 
+ It would print the total reward while training. Maybe I should use TensorBoardX?
+ There are many comment in the code. I believe these comments can answer some of your questions.

### How to use other DRL?
For example, if you want to use SAC, do the following: (I don't like import argprase)

1. See `run__xxx()` in `AgentRun.py`.
2. Use `run__zoo()` to run an off-policy algorithm. Use `run__ppo()` to run on-policy such as PPO.
3. Choose a DRL algorithm: `from AgentZoo import AgentXXX`.
4. Choose a gym environment: `args.env_name = "LunarLanderContinuous-v2"`

### Model-free DRL Algorithms

![model-free_DRL_2020](https://github.com/Yonv1943/DL_RL_Zoo/blob/master/Result/model-free_DRL_2020.png)
You can find above figure in `./Result/model-free_DRL_2020.png` or `*.pdf`.
More Policy Gradient Algorithms (Actor-Critic Methods) see the follow:
+ [Policy Gradient Algorithms written by Lilian Weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
+ [Policy Gradient Algorithms translated by Ewan Li in Chinese](https://tomaxent.com/2019/04/14/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/)

---
## If Necip want me to translate the following Chinese into English. I will. 
---


## 为何我要写这个库？
Github上有很多深度强化学习无模型算法的开源库。为何我还要重复造轮子呢？

1. **首要原因：那些库需要非常多的依赖，用起来非常麻烦**，因此我开源的DRL库主体只有3个python文件，且依赖的第三方库极少（只需要pytorch 和 gym）。因此我说它 轻量 lightweight 实至名归。
2. **实现细节会影响算法的稳定性，在DRL中更甚**（论文不一定会写全这些细节）。众所周知，DRL算法比起其他DL算法不够稳定（就像2018年前的GAN一样不稳定）。我之前有做GAN的经验，并且我博采众长，谨慎地在基础流程中使用了一些方法去稳定训练。我绝不通过微调超参数在某个任务中取得高分，所有默认的超参数的选值都是稳定优先，不要轻易修改。因此我说它 稳定 stable。
3. 尽管一些入门代码用简单的代码实现了简单的功能，**然而其运行时间长，效率低**。而我在注重代码优雅的情况下（身体力行Python之父的Python之禅），竭力地保持了代码的运行效率。例如，我专门比较了简单实现下的几种replay buffer，然后择优选用。因此我说它 高效 efficient。



#### 为何我这么排斥用第三方库？

主要原因：当你使用OpenAI的baselines 或者 stable-baselines时，你需要安装许多第三方库，例如（待补充），甚至还要装 Theano。**我复现其他人的算法时深受其苦**，因此我的代码绝不要求使用者安装那么多东西。

次要原因：我知道用一些第三方库可以减少代码量，但是为了极致简约，我倾向于不用。又由于我是初学者（我只有深度学习图像和多线程的基础），因此**我对优秀的第三方库了解不足，非常需要你们的推荐**（先感谢推荐的人）。

#### 为何我选用PyTorch 而非 TensorFlow？
1. TensorFlow 1.2 ~ 1.9 （静态图）**我用过很长一段时间**，后来发现科研的 PyTorch （动态图）节省时间。
3. TensorFlow 2 的 tf.keras 喧宾夺主。
4. TensorFlow 2 抛弃了 TensorFlow 1 静态图的性能优势。
2. TensorFlow 2 也用了动态图。然其同一功能，多种实现。文档混乱，朝令夕改。

综上，我彻底抛弃TensorFlow 2。
而不选用TensorFlow 1 而选用PyTorch是因为：

1. 调试简单，做科研需要写新东西；代码风格稳定，不用重复学习。
2. 代码可读性高，更像是Python的内置可微分Numpy。
3. 尽管 PyTorch 0.4 运行速度慢于TensorFlow 1（在2019年我做过测试）。但是现在PyTorch 1.x 已经快很多了。

### DRL 算法的稳定性
我非常清楚微调超参数可以在实验室里刷出很高的分数，用这些方法可以为某个算法得到虚高的分数。然而无论是学术界抑或工业界，事实上我们需要的是一个泛用的，稳定的算法。
我坚持以下原则：

1. 绝不使用需要精心调试的 OU Process. 
2. 绝不使用一些随着训练次数增加而变小的参数。例如噪声方差逐渐减小，我绝不用。
3. 使用网格搜索 grid search 从一堆超参数中选出合适的一组。如果这组参数的泛用性越高，则其价值越高。我将默认的超参数写在 `class Argument() in AgentRun.py`。里面的很大超参数都是 `2 ** n`，这就是为了表明我只对超参数进行粗调整，我绝不使用某些奇奇怪怪的值。



# More

The following content is too old and need to update.
以下内容比较旧，需要更新。

If you can understand Chinese, more details of DelayDDPG are described in Chinese in this website ↓
 
[Enhanced Learning DelayDDPG, LunarLander, BipedalWalker, only need to train half an hour of lightweight, stable code](https://zhuanlan.zhihu.com/p/72586697)

如果你能看得懂中文，那么我用中文写了对这个算法的详细介绍:
  
[强化学习DelayDDPG，月球着陆器，双足机器人，只需训练半个小时的轻量、稳定代码](https://zhuanlan.zhihu.com/p/72586697)
 
